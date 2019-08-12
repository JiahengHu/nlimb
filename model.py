import tensorflow as tf
import numpy as np
from deeplearning import tf_util as U, layers, module
from deeplearning.distributions import DiagGaussianMixturePd, CategoricalPd, BernoulliPd, DiagGaussianPd
from rl.rl_module import Policy, ActorCritic, ValueFunction

class GmmPd(DiagGaussianMixturePd):
    """
    Change GMM model to only allow gradients
    through the sampled component.
    """
    def sample(self):
        samples = tf.stack([g.sample() for g in self.gaussians])
        m = self.mixture.sample()
        s = tf.concat([tf.gather(samples, m)[0], tf.cast(m, tf.float32)[None]], axis=1)
        return s

    def mode(self):
        modes = tf.stack([g.mode() for g in self.gaussians])
        logps = tf.stack([g.logp(g.mode()) + self.log_mixing_probs[:,i] for i,g in enumerate(self.gaussians)])
        m = tf.argmax(logps)
        s = tf.concat([tf.gather(modes, tf.argmax(logps))[0], tf.cast(m, tf.float32)[None]], axis=1)
        return s

    def neglogp(self, x):
        params = x[:,:-1]
        comp = tf.cast(x[:,-1:], tf.int32)
        comp = tf.concat([comp, tf.expand_dims(tf.range(comp.shape[0]),axis=1)], axis=1)
        p = tf.stack([self.log_mixing_probs[:,i] + self.gaussians[i].logp(params) for i in range(self.n)])
        p = tf.gather_nd(p, comp)
        return -1. * p

#change GmmPd to add the connection list functionality
class GmmPdWithStruct(GmmPd):
    def __init__(self, flat, n, nlegs, hard_grad=False):
        self.n = n
        self.hard_grad = hard_grad
        self.nlegs = nlegs #this actually means the number of breakable joints
        self.mixture = CategoricalPd(flat[:,:n])
        self.log_mixing_probs = tf.nn.log_softmax(self.mixture.logits)
        self.gaussians = []
        self.bernoullis = []
        #we probably also need to change this
        #I don't really know why they write this as it is
        #this is 22, which is obviously wrong (should be...?)
        d = flat[:,n:].shape[1].value // n
        # print(f"d in model.py is {d}")
        for i in range(n):
            self.gaussians.append(DiagGaussianPd(flat[:,n+i*d:n+(i+1)*d-nlegs]))
            self.bernoullis.append(BernoulliPd(flat[:,n+(i+1)*d-nlegs:n+(i+1)*d]))

    def sample(self):
        samples = tf.stack([g.sample() for g in self.gaussians])
        bernoulli_samples = tf.stack([b.sample() for b in self.bernoullis])
        m = self.mixture.sample()
        #not exactly sure if I'm doing the correct thing here
        s = tf.concat([tf.gather(samples, m)[0], 
            tf.gather(bernoulli_samples, m)[0], tf.cast(m, tf.float32)[None]], axis=1)
        #should probably check the dimension of s
        return s

    def mode(self):
        modes = tf.stack([g.mode() for g in self.gaussians])
        bernoulli_modes = tf.stack([b.mode() for b in self.bernoullis])

        #what is this line doing?
        #the probability of getting the mode + the probability of the gmm?
        #might got dimensionality issue here, but we'll see
        temp_logs = [b.logp(b.mode()) + self.log_mixing_probs[:,i] for i,b in enumerate(self.bernoullis)]
        logps = tf.stack([g.logp(g.mode()) + temp_logs[i] for i,g in enumerate(self.gaussians)])
        #also add the prob of the nernoulli

        m = tf.argmax(logps)
        s = tf.concat([tf.gather(modes, tf.argmax(logps))[0], 
            tf.gather(bernoulli_modes, m)[0], tf.cast(m, tf.float32)[None]], axis=1)
        return s

    #now what does x represent? It should now be params + connection_list
    #what is comp? need to change this later
    #comp seems to be the m appended at the end
    def neglogp(self, x):
        params = x[:,:-1]
        #should be param+connection
        comp = tf.cast(x[:,-1:], tf.int32)
        comp = tf.concat([comp, tf.expand_dims(tf.range(comp.shape[0]),axis=1)], axis=1)
        #print(f"in model.py, shape for log mixing probs, gaussian and bernoullis is respectively 
        # print(self.log_mixing_probs[:,0].shape)
        # print(params[:,:-self.nlegs].shape)
        # print(self.gaussians[0].logp(params[:,:-self.nlegs]).shape)
        # print(self.bernoullis[0].logp(params[:,-self.nlegs:]).shape)
        p = tf.stack([self.log_mixing_probs[:,i] + self.gaussians[i].logp(params[:,:-self.nlegs]) + 
            self.bernoullis[i].logp(params[:,-self.nlegs:]) for i in range(self.n)])
        p = tf.gather_nd(p, comp)
        return -1. * p


class Net(module.Module):
    """
    Fully connected network.
    """
    ninputs=1
    def __init__(self, name, *modules, hiddens=[], activation_fn=tf.nn.tanh):
        super().__init__(name, *modules)
        self.hiddens = hiddens
        self.activation_fn = activation_fn

    def _build(self, inputs):
        net = tf.clip_by_value(inputs[0], -5.0, 5.0)
        for i,h in enumerate(self.hiddens):
            net = tf.layers.dense(
                net,
                units=h,
                kernel_initializer=U.normc_initializer(1.0),
                activation=self.activation_fn,
                name='dense{}'.format(i)
            )
        return net

#what exactly is this doing?
class RobotSampler(module.Module):
    """
    Define robot distribution.
    """
    ninputs=1
    def __init__(self, name, robot, nparams, ncomponents=8, mean_init=None, std_init=0.577, nlegs = 2, pleg_init = None):
        super().__init__(name, robot)
        self.nparams = nparams - nlegs  #here this parameter only represent those that are represented by Gaussian
        self.ncomponents = ncomponents
        self.mean_init = mean_init
        self.std_init = std_init
        self.nlegs = nlegs #this determines the number pf legs
        self.pleg_init = pleg_init #the initialy probability

    #not exacly sure where this function is called and what the inputs are referring to
    def _build(self, inputs):
        sampled_robot = inputs[0]
        vars = []
        vars.append(tf.get_variable('mixprobs',
                                     shape=(self.ncomponents,),
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=False))

        #randomly generate the starting distribution
        #self.pd is the eventual production
        #here is the problem, the parameters are treated as
        for i in range(self.ncomponents):
            if self.mean_init is not None:
                mean = np.asarray(self.mean_init[:-nlegs])
            else:
                mean = np.random.uniform(-0.8,0.8, size=self.nparams)
            m = tf.get_variable('m{}'.format(i),
                                shape=mean.shape,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(mean))
            logstd = np.log(self.std_init * np.ones_like(mean))
            s = tf.get_variable('logstd{}'.format(i),
                                shape=logstd.shape,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(logstd))
            vars.append(m)
            vars.append(s)

            #for each component, also create the probably for the legs
            #make sure that the number is between 0 and 1, check how they did that for the parameters
            if self.pleg_init is None:
                pleg = np.asarray(self.pleg_init)
            else:
                #do this if it is using logits (I'm not sure why)
                pleg = np.random.uniform(-1.0,1.0, size=self.nlegs)
            p = tf.get_variable('plegs{}'.format(i),
                                shape=pleg.shape,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(pleg))
            #this p is the variable that will be trained, and used to generate the bernoulli
            #should be two dimensional
            vars.append(p)

        gmm_params = tf.tile(tf.expand_dims(tf.concat(vars, axis=0), axis=0), [self.nbatch*self.nstep, 1])
        #(flat, n)
        self.pd = GmmPdWithStruct(gmm_params, self.ncomponents, self.nlegs)

        #added part, generate the connection_list


        #sample a component
        self._sample_component = self.pd.mixture.sample()
        #sample for each of the gassian distribution, seems to be tf_placeholder
        self._sample_gaussians = [g.sample() for g in self.pd.gaussians]
        self._sample_bernoullis = [g.sample() for g in self.pd.bernoullis]
        self._sample_params = tf.concat([self._sample_gaussians,self._sample_bernoullis], axis = 2)
        #best_model_generator
        self._mode = self.pd.mode()
        #sampler
        self._sample = self.pd.sample()
        return self.pd.neglogp(sampled_robot)

    #here concate it with the connection_list
    def sample(self, stochastic=True):
        if not stochastic:
            return self._mode.eval()
        else:
            return self._sample.eval()

    def sample_component(self):
        return self._sample_component.eval()

    #changed so that it samples both gaussian and bernoulli
    def sample_gaussian(self, index):
        s = self._sample_params[index].eval()
        #print(f"sample_params in model.py: {s} (this should be called in chopping process)")
        return np.concatenate([s, [[index]]], axis=1)


class RunningObsNorm(layers.RunningNorm):
    """
    Only normalize observations, not robot params.
    """
    ninputs=1
    def __init__(self, name, *modules, param_size=None):
        assert param_size is not None
        self.size = param_size
        super().__init__(name, *modules)

    def _build(self, inputs):
        X = inputs[0]
        obs = X[:,:-self.size]
        obs_normed = super()._build([obs])
        return tf.concat([obs_normed, X[:,-self.size:]], axis=-1)

    def update(self, mean, var, count):
        super().update(mean[:-self.size], var[:-self.size], count)

class Model(ActorCritic):
    """
    Combine policy, value function and robot distribution in one Module.
    """
    def __init__(self, name, policy, value_function, robot_sampler):
        super().__init__(name, policy, value_function)
        self.sampler = robot_sampler
