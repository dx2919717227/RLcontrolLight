import numpy as np
import random
from light_env import Light
import time
from collections import defaultdict

class QLearning(object):

    def __init__(self):
        self.lr = 0.7
        self.Q = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.memory = []
        self.qlEpsilon = 0.1
        self.qlDiscount = 0.8

    def learn(self, state, action, reward, state_next):
        state = self.getState(state)
        state_next = self.getState(state_next)
        cur_Q = self.Q[state][action]
        next_Q = reward + self.qlDiscount * max(self.Q[state_next])
        self.Q[state][action] += self.lr * (next_Q - cur_Q)

    def chooseAction(self, state):
        s_next = self.getState(state)
        if not s_next in self.Q:
            self.Q[s_next] = [0, 0, 0]
        if random.random() < self.qlEpsilon:
            a_next = random.choice(range(3))
        else:
            a_next = self.argMax(self.Q[s_next])
        return a_next

    def argMax(self, state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        print(max_index_list)
        return random.choice(max_index_list)

    def getState(self, state):
        [btsensor1, btsensor2, dissensor1, irsensor1, light_state] = state
        return str(btsensor1 // 100) + "_" + str(btsensor2 // 100)


    def save(self):
        pass

    def load(self):
        pass

if __name__ == "__main__":
    env = Light()
    agent = QLearning()
    bright_values = [25, 65, 100]
    for episode in range(1000):
        reward_sum = 0
        state = env.reset()
        while(True):
            action = agent.chooseAction(state)
            action_data = {
                "method": "set_bright",
                "value": bright_values[action],
                "changer": 1
            }
            state_next, reward, done, info = env.step(action_data)
            print(action_data, reward)
            # 先采用离散的方法，暂定3个档位
            if reward < 0:
                if info["bright"] <= 30 and action == 0:
                    reward = 1
                elif info["bright"] > 30 and info["bright"] <= 75 and action == 1:
                    reward = 1
                elif info["bright"] > 75 and info["bright"] <= 100 and action == 2:
                    reward = 1
            print(reward, state, state_next)
            reward_sum += reward
            print(action)
            agent.learn(state, action, reward, state_next)

