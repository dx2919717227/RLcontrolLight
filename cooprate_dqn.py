import os
import gym
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

class DQN(object) :
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')
        self.trained_model = self.trained_model()
        self.trained_model.load_weights("model/dqn.h5")
        self.model = self.build_model()

        # experience replay.
        self.memory_buffer = deque(maxlen=2000)
        # discount rate for q value.
        self.gamma = 0.95
        # epsilon of ε-greedy.
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01

    def load(self):
        if os.path.exists('model/cooperate_dqn.h5'):
            self.model.load_weights('model/cooperate_dqn.h5')

    def trained_model(self):
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=Adam(1e-3))

        return model

    def build_model(self):
        inputs = Input(shape=(6,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=Adam(1e-3))

        return model

    def egreedy_action(self, state):
        if np.random.rand() <= self.epsilon:
             return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self, batch):
        # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        # Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

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
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                x = observation.reshape(-1, 4)
                output_trained = self.trained_model.predict(x)[0]
                x_new = np.append(x[0], output_trained).reshape(-1, 6)
                action = self.egreedy_action(x_new)
                observation, reward, done, _ = self.env.step(action)
                reward_sum += reward
                output_trained_next = self.trained_model.predict(observation.reshape(-1, 4))[0]
                observation_new = np.append(observation, output_trained_next)
                self.remember(x_new[0], action, reward, observation_new, done)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # reduce epsilon pure batch.
                    if self.epsilon >= self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss,
                                                                                          self.epsilon))

        self.model.save_weights('model/cooperate_dqn.h5')

        return history

    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
    def plot(self, history):
        x = history['episode']
        r = history['Episode_reward']
        l = history['Loss']
        print(history)
        fig = plt.figure(figsize=(960,480))
        ax = fig.add_subplot(121)
        ax.plot(x, r)
        ax.set_title('Episode_reward')
        ax.set_xlabel('episode')
        ax = fig.add_subplot(122)
        ax.plot(x, l)
        ax.set_title('Loss')
        ax.set_xlabel('episode')

        plt.show()

    def play(self):
        print('play...')
        observation = self.env.reset()

        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()

            x = observation.reshape(-1, 4)
            output_trained = self.trained_model.predict(x)[0]
            x_new = np.append(x[0], output_trained).reshape(-1, 6)
            action = np.argmax(self.model.predict(x_new)[0])
            observation, reward, done, _ = self.env.step(action)

            reward_sum += reward

            if done:
                print("Reward for this episode was: {}".format(reward_sum))
                random_episodes += 1
                reward_sum = 0
                observation = self.env.reset()

        self.env.close()
if __name__ == '__main__':
    model = DQN()

    history = model.train(300, 32)
    model.save_history(history, 'dqn.csv')
    model.plot(history)