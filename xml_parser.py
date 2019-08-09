import numpy as np
import xml.etree.ElementTree as ET
import copy

#We need to change this file in order to really update the robot structure
#look into body class, as the tree is modified there

class Geom(object):
    def __init__(self, geom):
        self.xml = geom
        self.params = []

    def get_params(self):
        return self.params.copy()

    def set_params(self, new_params):
        self.params = new_params

    def update_point(self, p, new_params):
        pass

    def update_xml(self):
        pass

    def update(self, new_params):
        self.set_params(new_params)
        self.update_xml()

    def get_smallest_z(self):
        pass

    def get_param_limits(self):
        pass

    def get_param_names(self):
        pass

    def get_volume(self):
        pass

class Sphere(Geom):
    min_radius = .05
    max_radius = .4

    def __init__(self, geom):
        self.xml = geom
        self.params = [float(self.xml.get('size'))] # radius
        self.center = np.array([float(x) for x in self.xml.get('pos').split()])

    def update_point(self, p, new_params):
        return ((p - self.center) * new_params[0] / self.params[0]) + self.center

    def update_xml(self):
        self.xml.set('size', str(self.params[0]))

    def get_smallest_z(self):
        return self.center[2] - self.params[0]

    def get_param_limits(self):
        return [[self.min_radius], [self.max_radius]]

    def get_param_names(self):
        return ['radius']

    def get_volume(self):
        return 4./3. * np.pi * self.params[0] ** 3

class Capsule(Geom):
    min_length = 0.175
    max_length = 0.8
    min_radius = 0.035
    max_radius = 0.085

    def __init__(self, geom):
        self.xml = geom
        fromto = [float(x) for x in self.xml.get('fromto').split()]
        self.p1 = np.array(fromto[:3])
        self.p2 = np.array(fromto[3:])
        length = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        radius = float(self.xml.get('size'))
        self.params = [length, radius]
        self.axis = (self.p2 - self.p1) / length

    def update_point(self, p, new_params):
        lfac = p.dot(self.axis) * self.axis
        rfac = p - lfac
        return p + lfac * (-1.0 + new_params[0] / self.params[0])# + rfac * (new_params[1] / self.params[1])

    def update_xml(self):
        self.xml.set('fromto', ' '.join([str(x) for x in np.concatenate([self.p1, self.p2])]))
        self.xml.set('size', str(self.params[1])) # radius

    def set_params(self, new_params):
        #p1, p2 are the start and end position of the capsule
        p1 = self.update_point(self.p1, new_params)
        p2 = self.update_point(self.p2, new_params)
        # update only after computing p1, p2
        self.p1 = p1
        self.p2 = p2
        super().set_params(new_params)

    def get_smallest_z(self):
        return min(self.p1[2], self.p2[2]) - self.params[1]

    def get_param_limits(self):
        return [[self.min_length, self.min_radius], [self.max_length, self.max_radius]]

    def get_param_names(self):
        return ['length','radius']

    def get_volume(self):
        return 4./3. * np.pi * self.params[1]**3 + self.params[0] * np.pi * self.params[1]**2

#this seems to be controlling the robot
class Body:
    geoms = {'sphere': Sphere, 'capsule': Capsule} # dictionary of legal geometry types

    def __init__(self, body, worldbody=False):
        self.xml = body
        self.worldbody = worldbody
        # self.disabled_parts = [] #in the simplest case, we can have zero/one/two/three/four legs

        geom_xml = body.find('geom') # assume only one geometry per body
        self.geom = self.geoms[geom_xml.get('type')](geom_xml)
        self.joints = [j for j in body.findall('joint') if 'ignore' not in j.get('name')]
        self.parts = [Body(b) for b in body.findall('body')]
        pos = [b.get('pos') for b in body.findall('body')]
        self.part_positions = [np.array([float(x) for x in p.split()]) for p in pos]
        pos = [j.get('pos') for j in self.joints]
        self.joint_positions = [np.array([float(x) for x in p.split()]) for p in pos]
        self.n = len(self.geom.get_params())
        self.n_all_params = len(self.get_params())

        self.zmin = float(self.xml.get("pos").split()[2]) - self.get_height()

    #The maximum height of the model (solely based on the xml)? kinda doesn't make sense
    def get_height(self):
        #smallest z is just the smallest z value
        max_height = -self.geom.get_smallest_z()
        for body, pos in zip(self.parts, self.part_positions):
            max_height = max(max_height, body.get_height() - pos[2])
        return max_height

    #why do we even want to update the initial position?
    def update_initial_position(self):
        pos = self.xml.get("pos").split()
        pos[2] = str(self.get_height() + self.zmin)
        self.xml.set("pos", ' '.join(pos))

    #given a set of body and joint pos, update the position of them
    def update_xml(self):
        for body, pos in zip(self.parts, self.part_positions):
            body.xml.set('pos', ' '.join([str(x) for x in pos]))

        for joint, pos in zip(self.joints, self.joint_positions):
            joint.set('pos', ' '.join([str(x) for x in pos]))

    #this function: 1. it seems to update the "fromto of the capsul"
    def set_body_positions(self, new_params):
        for i, pos in enumerate(self.part_positions):
            self.part_positions[i] = self.geom.update_point(pos, new_params)
        for i, pos in enumerate(self.joint_positions):
            self.joint_positions[i] = self.geom.update_point(pos, new_params)

    #what exactly does this update do?
    def update(self, new_params):
        self.set_body_positions(new_params)
        self.geom.update(new_params)
        self.update_xml()

    def get_params(self):
        params = self.geom.get_params()
        for body in self.parts:
            params += body.get_params()
        return params

    def get_param_limits(self):
        limits = self.geom.get_param_limits()
        for body in self.parts:
            body_limits = body.get_param_limits()
            limits[0] += body_limits[0]
            limits[1] += body_limits[1]
        return limits

    def get_param_names(self):
        name = self.xml.get('name')
        param_names = [name + '-' + p for p in self.geom.get_param_names()]
        for body in self.parts:
            param_names += body.get_param_names()
        return param_names

    def update_params(self, new_params):
        if self.worldbody: assert len(new_params) == self.n_all_params, "Wrong number of parameters"
        self.update(new_params[:self.n])
        remaining_params = new_params[self.n:]
        for body in self.parts:
            remaining_params = body.update_params(remaining_params)
        if self.worldbody:
            self.update_initial_position()
        else:
            return remaining_params

    def get_body_names(self):
        names = [self.xml.get('name')]
        for body in self.parts:
            names += body.get_names()
        return names

    def get_joints(self):
        joints = {}
        for body,pos in zip(self.parts, self.part_positions):
            for j in body.joints:
                joints[j.get('name')] = (self.xml.get('name'), body.xml.get('name'), self.geom, body.geom, pos)
            joints.update(body.get_joints())
        return joints

    def get_volumes(self):
        volumes = {}
        if len(self.joints) > 0:
            for j in self.joints:
                v1 = self.geom.get_volume()
                v2 = sum([b.geom.get_volume() for b in self.parts])
                volumes[j.get('name')] = np.array((v1, v2))
        for body in self.parts:
            volumes.update(body.get_volumes())
        return volumes

    # #newly added function to update the structure of the mujoco xml
    # #maybe we can keep this and just create a new body everytime we sample a robot
    # #if we only change the body and assign random pos, the update param should be able to 
    # #take care of the actual pos
    # def update_struct(self, connection_list):
    #     tmp_xml = copy.deepcopy(self.xml)
    #     self.ori_xml = self.xml
    #     self.xml = tmp_xml
    #     #problem: here the two xml are actually different

    #     num_total_parts = len(self.parts)
    #     print(f"total part number is {num_total_parts}")
    #     parts_xml = self.xml.findall('body')
    #     for i in range(num_total_parts):
    #         j = num_total_parts - i
    #         if j not in connection_list:
    #             self.xml.remove(parts_xml[j-1])
    #             #self.xml.remove(self.joints[j-1])
    #             # self.parts.pop([j])
    #             # self.joints.pop([j])

    #     #we probably need to recall init after this just to reset everything

    # def revert_model(self):
    #     self.xml = self.ori_xml

    #but do we really want to do this? this will change the n_all_params which we might not really want
    def reset(self):
        self.joints = [j for j in body.findall('joint') if 'ignore' not in j.get('name')]
        self.parts = [Body(b) for b in body.findall('body')]
        pos = [b.get('pos') for b in body.findall('body')]
        self.part_positions = [np.array([float(x) for x in p.split()]) for p in pos]
        pos = [j.get('pos') for j in self.joints]
        self.joint_positions = [np.array([float(x) for x in p.split()]) for p in pos]
        self.n = len(self.geom.get_params())
        self.n_all_params = len(self.get_params())

        self.zmin = float(self.xml.get("pos").split()[2]) - self.get_height()




#maybe we also need to modify this, I'm not sure
#(Need deeper understanding of the code)
class MuJoCoXmlRobot:
    def __init__(self, model_xml):
        self.model_xml = model_xml
        self.tree = ET.parse(self.model_xml)
        worldbody = self.tree.getroot().find('worldbody')
        self.body = Body(worldbody.find('body'), worldbody=True)
        #during update, we should probably just complete change self.body
        #we can think of self.body as the skeleton while having a list that controls
        #how many legs will be incorporated

    def get_params(self):
        return self.body.get_params()

    def get_param_limits(self):
        return self.body.get_param_limits()

    def get_param_names(self):
        return self.body.get_param_names()

    def get_height(self):
        return self.body.get_height()

    def get_joints(self):
        return self.body.get_joints()

    def get_volumes(self):
        return self.body.get_volumes()


    #this update would now not only be updating the parameters, but also the body
    def update(self, params, xml_file=None, connection_list = [1,2,3,4]):
        if xml_file is None:
            xml_file = self.model_xml
        self.body.update_params(list(params))
        self.tree.write(xml_file)



        #also update the actuator

    #this will be the way to go if we want to create from scratch, which will probably be what we will 
    #eventually do
    #update the body structure
    # def update_struct(self, adj_matrix):
    #     worldbody = ET.Element('worldbody')
        
    #     # parse the matrix
    #     N, _ = adj_matrix.shape

    #     body_dict = {}
    #     info_dict = {} # log the needed information given a node
        
    #     # the root of the model is always fixed
    #     body_root = ET.Element('body', name='torso', pos='0 0 0.75')
    #     torso_geom = etree.Element('geom', name="torso", type="sphere", size=".25", pos="0 0 0")
    #     body_root.append(torso_geom)

    #     root_info = {}

    #     # root_info['a_size'] = 0.01
    #     # root_info['b_size'] = 0.08
    #     # root_info['c_size'] = 0.04
    #     # root_info['abs_trans'] = homogeneous_transform(np.eye(3), np.zeros(3))
    #     # root_info['rel_trans'] = homogeneous_transform(np.eye(3), np.zeros(3))

    #     info_dict[0] = root_info
    #     body_dict[0] = body_root

    #     #changed till here

    #     # initilize the parent list to go throught the entire matrix
    #     parent_list = [0]
    #     while len(parent_list) != 0:
    #         parent_node = parent_list.pop(0)

    #         parent_row = np.copy(adj_matrix[parent_node])
    #         for i in range(parent_node+1): parent_row[i] = 0
    #         child_list = np.where(parent_row)[0].tolist()

    #         while True:
                
    #             try: child_node = child_list.pop(0)
    #             except: break

    #             #poping all the children connectied to the parent

    #             # parent-child relationship 
    #             # print('P-C relationship:', parent_node, child_node)
    #             node_attr = node_attr_list[child_node]
    #             node_name = 'node-%d'%(child_node)

    #             # this is parent's ellipsoid information
    #             parent_info = info_dict[parent_node]
    #             a_parent = parent_info['a_size']
    #             b_parent = parent_info['b_size']
    #             c_parent = parent_info['c_size']

    #             # randomly sample a point on ellipsoid
    #             u = node_attr['u']
    #             v = node_attr['v']
    #             x, y, z = model_gen_util.vectorize_ellipsoid(a_parent, b_parent, c_parent,
    #                                                          u, v)

    #             # use the normal vector as the child y-axis
    #             normal_vector = model_gen_util.ellipsoid_normal_vec(a_parent, b_parent, c_parent, 
    #                                                                 np.array([x, y, z]))
    #             y_axis_vec = normal_vector
    #             # according to the y-axis, sample an x-axis,
    #             # the two vectors of x and y axis dot product need to be zero, fix x, y -> get z 
    #             axis_x = node_attr['axis_x']
    #             axis_y = node_attr['axis_y']
    #             axis_z = (normal_vector[0] * axis_x + normal_vector[1] * axis_y) / (-normal_vector[2] + 1e-8)
    #             x_axis_vec = np.array([axis_x, axis_y, axis_z])
    #             x_axis_vec = x_axis_vec / (np.linalg.norm(x_axis_vec) + 1e-8)

    #             # use cross-product (x-axis cross-product y-axis) (according to right-hand rule) 
    #             # find the z-axis 
    #             z_axis_vec = np.cross(x_axis_vec, y_axis_vec)

    #             # WARN: this is not euler angle even by x-y-z rotation
    #             # calculate the xyz euler angle, project onto the original frame vector and find the angle
    #             # this part is not used because of the usage of xyaxes when dealing with frame orientation in mujoco
    #             # theta_x = angle_between(x_axis_vec, np.array([1, 0, 0]))[0]
    #             # theta_y = angle_between(y_axis_vec, np.array([0, 1, 0]))[0]
    #             # theta_z = angle_between(z_axis_vec, np.array([0, 0, 1]))[0]
    #             # euler_angle = '%d %d %d' % (int(theta_x), int(theta_y), int(theta_z))
                
    #             a_child = node_attr['a_size']
    #             b_child = node_attr['b_size']
    #             c_child = node_attr['c_size']
    #             # the translation vector: moving from center to the randomly sampled point
    #             # and move along the z-direction with length c_child
    #             d_trans = np.array([x, y, z]) + normal_vector * b_child

    #             # compute the translational and rotational matrix
    #             child_info = {}
    #             translation_vec = d_trans
    #             ''' this part won't be used because of the way we set up rotation coordinates
    #             also, the transformation matrices are not needed at this stage
    #             '''
    #             # R_x = rotation_matrix('x', theta_x)  
    #             # R_y = rotation_matrix('y', theta_y)
    #             # R_z = rotation_matrix('z', theta_z)
    #             # try: R = np.matmul( np.matmul(R_z, R_y), R_x )
    #             # except: pdb.set_trace()
    #             # 
    #             # child_info['rel_trans'] = homogeneous_transform(R, translation_vec)
    #             # child_info['abs_trans'] = np.matmul( info_dict[parent_node]['abs_trans'], child_info['rel_trans'] )


    #             # store attributes that defines the child's geom
    #             child_info['a_size'] = a_child
    #             child_info['b_size'] = b_child
    #             child_info['c_size'] = c_child
                
    #             # body translation
    #             dx, dy, dz = translation_vec.tolist() 
    #             body_pos = '%f %f %f' % (dx, dy, dz)
    #             # joint_pos = np.matmul(child_info['rel_trans'], homogeneous_representation(np.array([x, y, z])))[0:3]
    #             # joint_x, joint_y, joint_z = joint_pos.tolist()
    #             joint_pos = '%f %f %f' % (0, -b_child, 0)
    #             x_axis_x, x_axis_y, x_axis_z = x_axis_vec.tolist()
    #             y_axis_x, y_axis_y, y_axis_z = y_axis_vec.tolist()
                
    #             xyaxes = '%f %f %f %f %f %f' % (x_axis_x, x_axis_y, x_axis_z, y_axis_x, y_axis_y, y_axis_z)

    #             # now create the body
    #             body_child = etree.Element('body', name=node_name, pos=body_pos, xyaxes=xyaxes)
    #             # add geom
    #             geom_type = node_attr['geom_type']
    #             if geom_type == 0:   # ellipsoid
    #                 ellipsoid_size = '%f %f %f' % (a_child, b_child, c_child)
    #                 geom = etree.Element('geom', name=node_name, type='ellipsoid', size=ellipsoid_size)
    #             elif geom_type == 1: # cylinder
    #                 cylinder_size = '%f %f' % (geom_size, geom_size * CYLINDER_H_RATIO / 2)
    #                 geom = etree.Element('geom', name=node_name, type='cylinder', size=cylinder_size)
    #             else:
    #                 from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
    #                 raise RuntimeError('geom type not supported')
    #             body_child.append(geom)
                
    #             # add joints
    #             joint_type = adj_matrix[parent_node, child_node]
    #             joint_axis = [int(item) for item in list(bin(joint_type)[2:])]
    #             joint_axis = [0 for i in range(3-len(joint_axis))] + joint_axis
    #             joint_axis = list(reversed(joint_axis))
    #             joint_range = node_attr['joint_range']
    #             if joint_axis[0] == 1:
    #                 x_joint = etree.fromstring("<joint name='%d-%d_x' axis='1 0 0' pos='%s' range='-%d %d'/>" % (parent_node, child_node, joint_pos, joint_range, joint_range))
    #                 body_child.append(x_joint)
    #             if joint_axis[1] == 1:
    #                 y_joint = etree.fromstring("<joint name='%d-%d_y' axis='0 1 0' pos='%s' range='-%d %d'/>" % (parent_node, child_node, joint_pos, joint_range, joint_range))
    #                 body_child.append(y_joint)
    #             if joint_axis[2] == 1:
    #                 z_joint = etree.fromstring("<joint name='%d-%d_z' axis='0 0 1' pos='%s' range='-%d %d'/>" % (parent_node, child_node, joint_pos, joint_range, joint_range))
    #                 body_child.append(z_joint)

    #             site = etree.Element('geom', type='sphere', pos=joint_pos, size='0.003', material='target')
    #             body_child.append(site)
                
    #             body_dict[parent_node].append(body_child)
    #             body_dict[child_node] = body_child # register child's body struct in case it has child
    #             info_dict[child_node] = child_info
    #             # child becomes the parent for further examination
    #             parent_list.append(child_node)

if __name__ == '__main__':
    robot = MuJoCoXmlRobot('mujoco_assets/ant.xml')
    params = [.2, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06]
    robot.update(params, 'mujoco_assets/ant_test.xml')
    assert robot.get_params() == params
    assert robot.get_height() == .2
    print(robot.get_param_limits())
    print(robot.get_param_names())
    robot.update(params, 'mujoco_assets/ant_test.xml', connection_list = [1])
    assert robot.get_params() == params
    assert robot.get_height() == .2
    print(robot.get_param_limits())
    print(robot.get_param_names())

    import gym, roboschool
    env = gym.make("RoboschoolAnt-v1")
    import os

    env.unwrapped.model_xml = os.path.join(os.getcwd(), 'mujoco_assets/ant_test.xml')
    #change footlist, change action space
    #these are just temp fix
    env.unwrapped.foot_list = ["front_left_leg"]
    high = np.ones([2])
    env.unwrapped.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
    env.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
    high = np.inf*np.ones([13])
    env.unwrapped.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
    env.reset()
    #env.render()
    import os
    from scipy.misc import imsave
    import subprocess as sp
    outdir = 'xml_vid'
    os.makedirs(outdir, exist_ok=True)
    i = 0
    for _ in range(10):
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            rgb = env.render('rgb_array')
            imsave(os.path.join(outdir, '{:05d}.png'.format(i)), rgb)
            i+=1
    sp.call(['ffmpeg', '-r', '60', '-f', 'image2', '-i', os.path.join(outdir, '%05d.png'), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(outdir, 'out.mp4')])
    env.close()

    exit(0)

    ####################################### the rest won't be accessed ##############################

    robot = MuJoCoXmlRobot('mujoco_assets/hopper.xml')
    params = list(1.0 * np.array(robot.get_params()))
    robot.update(params, 'mujoco_assets/hopper_test.xml')
    assert robot.get_params() == params
    #assert robot.get_height() == 1.31
    print(robot.get_param_limits())
    print(robot.get_param_names())

    robot = MuJoCoXmlRobot('mujoco_assets/walker2d.xml')
    params = [.4,.04,.5,.05,.55,.055,.6,.06,.5,.05,.55,.055,.6,.06]
    robot.update(params, 'mujoco_assets/walker2d_test.xml')
    assert robot.get_params() == params
    assert robot.get_height() == 1.31
    print(robot.get_param_limits())
    print(robot.get_param_names())

    robot = MuJoCoXmlRobot('mujoco_assets/ant.xml')
    params = [.2, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06]
    robot.update(params, 'mujoco_assets/ant_test.xml')
    assert robot.get_params() == params
    assert robot.get_height() == .2
    print(robot.get_param_limits())
    print(robot.get_param_names())

    robot = MuJoCoXmlRobot('mujoco_assets/humanoid.xml')
    params = list(.8 * np.array(robot.get_params()))
    robot.update(params, 'mujoco_assets/humanoid_test.xml')
    assert robot.get_params() == params
    print(robot.get_height())
    #assert robot.get_height() == .6085
    print(robot.get_param_limits())
    print(robot.get_param_names())

    import gym, roboschool
    env = gym.make("RoboschoolHopper-v1")
    env.unwrapped.model_xml = 'mujoco_assets/hopper_test.xml'
    env.reset()
    #env.render()
    import os
    from scipy.misc import imsave
    import subprocess as sp
    outdir = 'xml_vid'
    os.makedirs(outdir, exist_ok=True)
    i = 0
    for _ in range(10):
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            rgb = env.render('rgb_array')
            imsave(os.path.join(outdir, '{:05d}.png'.format(i)), rgb)
            i+=1
    sp.call(['ffmpeg', '-r', '60', '-f', 'image2', '-i', os.path.join(outdir, '%05d.png'), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(outdir, 'out.mp4')])
    env.close()
