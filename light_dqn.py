import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from light_env import Light


class DQN(object):
    def __init__(self):
        self.env = Light()
        self.model = self.build_model()
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.memory_buffer = deque(maxlen=2000)

        self.gamma = 0.95

    def load(self):
        pass
    # 模型输入为两个光传感器的值，是否有人，时间段（一天暂定分24个段）
    def build_model(self):
        inputs = Input(shape=(2, ))
        x = Dense(4, activation='relu')(inputs)
        x = Dense(4, activation='relu')(x)
        x = Dense(3, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=Adam(1e-3))
        return model

    def egreedy_action(self, state):
        if np.random.rand() <= self.epsilon:
             return random.randint(0, 2)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 从 replay buffer 中随机取出一组数据进行训练，并计算 td-error
    def process_batch(self, batch):
        data = random.sample(self.memory_buffer, batch)

        states = np.array([d[0] for d in data])
        next_states = np.array(d[3] for d in data)
        y = self.model.predict(states)
        q = self.model.predict(next_states)
        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y

    def train(self, episode, batch):
        history = {'episode': [], 'Episode_reward': [], 'Loss': []}
        count = 0
        action_data = {
            "method": "toggle",  # set_bright set_ct toggle
            "value": "on",
            "changer": 1
        }
        bright_values = [0, 50, 100]
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False
            while not done:
                x = self.process_observation(observation).reshape(-1, 2)
                action = self.egreedy_action(x)
                action_data = {
                    "method": "set_bright",
                    "value": bright_values[action],
                    "changer": 1
                }
                observation, reward, done, info = self.env.step(action_data)

                # 先采用离散的方法，暂定3个档位
                if reward < 0:
                    if info["bright"] <= 25 and action == 0:
                        reward = 1
                    elif info["bright"] > 25 and info["bright"] <= 75 and action == 1:
                        reward = 1
                    elif info["bright"] > 75 and info["bright"] <= 100 and action == 2:
                        reward = 1

                reward_sum += reward
                print(x, action)
                self.remember(x[0], action, reward, self.process_observation(observation), done)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)
                    count += 1
                    self.update_epsilon()

                # if i % 5 == 0:
                #     history['episode'].append(i)
                #     history['Episode_reward'].append(reward_sum)
                #     history['Loss'].append(loss)
                #
                #     print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss,
                #                                                        self.epsilon))
        self.model.save_weights('model/dqn.h5')

    def process_observation(self, observation):
        [btsensor1, btsensor2, dissensor1, irsensor1, light_state] = observation
        return np.array([btsensor1, btsensor2])

if __name__ == "__main__":
    model = DQN()
    history = model.train(100, 16)
