import numpy as np

class robot_control:
    def __init__(self, frame = 100):
        self.max_length = frame * 5
        self.action = np.full((self.max_length), None)
        self.action_time = np.full((self.max_length), None)
        self.current_index = 0
        self.is_full = False
        self.currentAction = "防禦"
        self.actions = ["防禦", 
                        "前進", 
                        "拳", 
                        "重拳", 
                        "腳", 
                        "重腳",
                        "上鉤拳",
                        "波動拳",
                        "連踢",
                        "連打"]
        self.charging = [0, 0, 0, 0]
    def eeg_input(self, eeg):
        action_num = None
        indices = np.nonzero(eeg)[0]
        if np.sum(eeg) == 1:
            action_num = indices[0]
        elif np.sum(eeg) == 2:
            first, second = indices[0], indices[1]
            if( first == 1 and second == 3):
                action_num = 6
            elif( first == 2 and second == 3):
                action_num = 7
            elif( first == 1 and second == 4):
                action_num = 8
            elif( first == 1 and second == 2):
                action_num = 9
            else:
                action_num = None
                # print("無法辨識")
        else:
            action_num = None
            # print("無法辨識")
        if action_num is not None:
            # print("This time action is",self.actions[action_num])
            self.add_action(action_num)

    def add_action(self, action_num):
        self.action[self.current_index] = action_num
        self.current_index = (self.current_index + 1) % self.max_length
        if self.current_index == 0:
            self.is_full = True
        if self.is_full:
            self.action_time = np.concatenate((self.action[self.current_index:], self.action[:self.current_index]))
            self.set_action()
        else:
            print("Robot still standby.")

    def set_action(self):
        action_time_array = np.array(self.action_time, dtype=int)
        most_common_action = np.argmax(np.bincount(action_time_array))
        self.currentAction = most_common_action

        last_action = self.action_time[-2]
        current_action = self.action_time[-1]
        if (last_action == 1 and current_action == 2) or current_action == 6:
            self.charging[0] += 1
        elif (last_action == 1 and current_action == 3) or current_action == 7:
            self.charging[1] += 1
        elif (last_action == 1 and current_action == 4) or current_action == 8:
            self.charging[2] += 1
        elif (last_action == 2 and current_action == 3) or current_action == 9:
            self.charging[3] += 1
        
        if self.charging[0] >= 300:
            self.charging[0] = 0
            self.currentAction = 6
            print("上鉤拳")
        elif self.charging[1] >= 300:
            self.charging[1] = 0
            self.currentAction = 7
            print("連打")
        elif self.charging[2] >= 300:
            self.charging[2] = 0
            self.currentAction = 8
            print("連踢")
        elif self.charging[3] >= 300:
            self.charging[3] = 0
            self.currentAction = 9
            print("波動拳")
        # print("The most current action is",self.currentAction)


    def get_action(self):
        return self.currentAction

def generate_one_hot(length):
    index = np.random.randint(0, length)
    index2 = np.random.randint(0, length)
    one_hot_array = np.zeros(length, dtype=int)
    one_hot_array[index] = 1
    one_hot_array[index2] = 1
    return one_hot_array

frame = 100
robot = robot_control()

for i in range(frame * 100):
    eeg_matrix = generate_one_hot(6)
    robot.eeg_input(eeg_matrix)