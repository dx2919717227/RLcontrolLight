import requests
import json
import pymysql
import datetime
import time

class Light(object):
    def __init__(self):
        self.user_operation_offset = 5 # 用户操作与决策之间时间间隔
        self.positive_reward = 1    # 用户未调整灯
        self.negative_reward = -10  # 用户对灯进行了调整
        self.positive_reward_decay = 0.01
        self.tmp_positive_reward = self.positive_reward
        self.url = "localhost:9090/lights/1"
        self.ip = "0.0.0.0"
        self.state = 0
        self.bright = 0
        self.ct = 0
        self.btsensor1 = 0
        self.dissensor1 = 0
        self.irsensor1 = 0
        self.count_step = 0
        self.max_step = 200
        self.env_buffer = []
        self.db = pymysql.connect("localhost", "root", "wangyulin", "light")
        self.cursor = self.db.cursor()

    def fetch_data(self):
        response = requests.get("http://localhost:9090/lights/1")
        data = json.loads(response.text)
        return data

    def get_state(self):
        data = self.fetch_data()
        btsensor1 = data['btsensor1']
        btsensor2 = data['btsensor2']
        dissensor1 = data['dissensor1']
        irsensor1 = data['irsensor1']
        light_state = 1 if data['state'] == 'on' else 0
        return [btsensor1, btsensor2, dissensor1, irsensor1, light_state]

    def step(self, action):
        requests.post("http://localhost:9090/lights/1", data=action)
        time.sleep(3)
        state = self.get_state()
        reward = self.get_reward()
        done = True if self.count_step == self.max_step else False
        self.count_step += 1
        info = {}
        if reward == self.negative_reward:
            time.sleep(1)
            data = self.fetch_data()
            info = {
                "bright": data['bright']
            }

        return state, reward, done, info       # 返回环境state，reward，是否结束

    def get_reward(self):
        time.sleep(self.user_operation_offset / 2)
        user_action = self.check_user_operation()
        if user_action == -1:                   #用户未调整灯
            self.tmp_positive_reward -= self.positive_reward_decay
            return self.tmp_positive_reward
        else:
            self.tmp_positive_reward = self.positive_reward
            return self.negative_reward

    def reset(self):
        self.db.close()
        self.__init__()
        state = self.get_state()
        return state

    def check_user_operation(self):
        self.cursor.execute('select * from actions where action = "bright" and changer=0 order by id desc limit 1')
        data = self.cursor.fetchone()
        create_time = data[2]
        time = datetime.datetime.now()
        time_diff = (time - create_time).seconds
        if time_diff < self.user_operation_offset:
            return data[5]
        else:
            return -1


if __name__ == "__main__":
    data = {
        "method": "set_bright",  # set_bright set_ct toggle
        "value": 50,
        "changer": 1
    }
    # data = {
    #     "method": "toggle",  # set_bright set_ct toggle
    #     "value": "on",
    #     "changer": 1
    # }
    light = Light()
    print(light.step(data))
