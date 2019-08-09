import numpy as np
from xml_parser import MuJoCoXmlRobot

def get_default_xml(robot_type):
    if robot_type == 'hopper':
        return 'assets/hopper.xml'
    elif robot_type == 'walker':
        return 'assets/walker2d.xml'
    elif robot_type == 'ant':
        return 'assets/ant.xml'
    else:
        assert False, "Unknown robot type."

def get_robot(robot_type):
    if 'hopper' == robot_type:
        return Hopper
    if 'walker' == robot_type:
        return Walker
    if 'ant' == robot_type:
        return Ant
    if 'walkerevo' == robot_type:
        return WalkerEvo
    assert False, "unkown robot"


# Add parameter constraints for the different robots
class Hopper(MuJoCoXmlRobot):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        self.default_robot = MuJoCoXmlRobot(get_default_xml('hopper'))
        self.default_params = np.array(self.default_robot.get_params())
        self.lower_limits = 0.5 * self.default_params
        self.upper_limits = 1.5 * self.default_params

    def get_param_limits(self):
        return self.lower_limits, self.upper_limits


class Ant(MuJoCoXmlRobot):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        self.default_robot = MuJoCoXmlRobot(get_default_xml('ant'))
        self.default_params = np.array(self.default_robot.get_params())
        self.lower_limits = 0.5 * self.default_params
        self.upper_limits = 1.5 * self.default_params

    def get_param_limits(self):
        return self.lower_limits, self.upper_limits

class Walker(MuJoCoXmlRobot):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        self.default_robot = MuJoCoXmlRobot(get_default_xml('walker'))
        self.default_params = np.array(self.default_robot.get_params()[:8])
        self.lower_limits = 0.5 * self.default_params
        self.upper_limits = 1.5 * self.default_params

    def get_params(self):
        return super().get_params()[:8]

    def get_param_limits(self):
        return self.lower_limits, self.upper_limits

    def get_param_names(self):
        return super().get_param_names()[:8]

    def update(self, params, xml_file=None):
        params = np.array(params)
        params = np.concatenate([params, params[2:]])
        super().update(params, xml_file)

#need to take care of the parameter limits and etc
class WalkerEvo(Walker):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        #here in this class we are actually making the assumption that 
        #foot list is the number of joints that can be disconnected, which might not always be the case
        self.foot_list = ["foot", "foot_left"]
        self.action_space_coeff = 3
        self.observation_space_coeff = 7
        self.connection_list = np.ones(len(self.foot_list))

    def get_param_limits(self):
        return np.concatenate(self.lower_limits, np.zeros(len(self.foot_list))), np.concatenate(self.upper_limits, np.ones(len(self.foot_list)))

    def get_params(self):
        return np.concatenate([super().get_params()[:8], self.connection_list])

    def get_param_names(self):
        return np.concatenate([super().get_param_names()[:8], self.foot_list])

    def update(self, params, xml_file=None, connection_list = [1,1]):
        if xml_file is None:
            xml_file = self.model_xml
        self.body.update_params(list(params))
        # self.body.update_struct(connection_list)
        # print("number of body")
        # print(self.body.xml.findall("body"))
        temp_tree = copy.deepcopy(self.tree)
        self.update_struct(connection_list)

        self.tree.write(xml_file)
        #most likely because self.tree has never been modified
        #self.body.revert_model()
        self.tree = temp_tree

    def update_struct(self, connection_list):
        #this update the structure of self.tree
        self.connection_list = connection_list
        xml = self.tree.getroot().find('worldbody').find('body')
        parts_xml = xml.findall('body')
        num_total_parts = len(parts_xml)
        actuator = self.tree.getroot().find('actuator')
        motors = actuator.findall('motor')

        print(f"total part number is {num_total_parts}")
        parts_xml = xml.findall('body')
        for i in range(num_total_parts):
            j = num_total_parts - i
            if connection_list[j-1] == 0:
                xml.remove(parts_xml[j-1])
                #this is amazingly inefficient and relies on the sequence of the actuator
                #but it works for the walker
                actuator.remove(motors[3*j-1])
                actuator.remove(motors[3*j-2])
                actuator.remove(motors[3*j-3])