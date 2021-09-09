from gym import error, spaces
from gym import Env
import random, os
import multiprocessing

import numpy as np
import pandas as pd

class Stock(Env):
    def __init__(self, code="005930", window_size=20, test=False, verbose=False):
        self.verbose = verbose

        # 오늘 이전으로 며칠동안 데이터를 관찰할 것인지를 window_size에 담음
        self.window_size = window_size
        self.action_size = 1

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_size,), dtype=np.float32)

        self.episode_count = 0

        # 30일간의 투자 결과를 지켜볼 예정
        self.episode_length = 30

        # Train 데이터와 Test 데이터를 분리
        if test == True:
            self.dataset_path = os.path.join(".","datasets","test",code+".csv")
        else:
            self.dataset_path = os.path.join(".","datasets","train",code+".csv")

        self.dataset = pd.read_csv(self.dataset_path)

    def reset(self):
        if self.verbose == True:
            print("-------------")
            print("Episode :", self.episode_count)
        # 모든 변수들을 초기화
        self.done = False

        self.step_count = 0

        self.balance = 10000000
        self.stock_amount = 0

        # 주어진 데이터셋에서 임의의 구간을 설정
        self.data = self.selectRange()

        self.episode_count += 1

        return self.obs()

    def step(self, action):
        # Action은 0~1 사이의 값으로 현재 (평가금액+잔액) 중 투자비율을 뜻함
        # e.g. 1 : 최대매수, 0.5 : 반절매수
        desired_stock_value = self.total_value("today") * action[0]
        current_stock_value = self.stock_value("today")

        # 오늘 주가
        today_info, _ = self.getCurrentScope()
        today_stock_price = today_info[-1]['Close']

        if desired_stock_value > current_stock_value:
            # 매수
            diff = desired_stock_value - current_stock_value
            amount = diff // today_stock_price

            pay_amount = amount * today_stock_price

            # 구매 희망금액보다 잔고가 적을 때
            if pay_amount > self.balance:
                amount = self.balance // today_stock_price
                pay_amount = amount * today_stock_price

            self.balance -= pay_amount
            self.stock_amount += amount

        elif desired_stock_value < current_stock_value:
            # 매도
            diff = current_stock_value - desired_stock_value
            amount = diff // today_stock_price

            # 주식 수가 모자를 경우
            if amount > self.stock_amount:
                amount = self.stock_amount
            
            refund_amount = amount * today_stock_price

            self.balance += refund_amount
            self.stock_amount -= amount

        reward = self.getReward()

        self.step_count += 1

        if self.verbose == True:
            print(self)

        # Episode length만큼 진행했으면 에피소드 종료
        if self.step_count == self.episode_length:
            self.done = True
            print("Episode: ", self.episode_count, "Total Value: ", self.total_value("tomorrow"))

        return self.obs(), reward, self.done, {'total_value': self.total_value("tomorrow")}

    def obs(self):
        obs_raw, _ = self.getCurrentScope()
        obs = [d['Close'] for d in obs_raw]

        # Normalize 목적으로 데이터를 나눔
        obs[:] = [x / 50000. for x in obs]

        return obs

    def getReward(self):
        # 현재의 평가금액과 잔고의 합을 리워드로 반환
        # 가지고만 있으면 안되기 때문에 Penalty를 부여
        return self.total_value("tomorrow") * (0.99 ** self.step_count) / 50000.

    def stock_value(self, flag):
        if flag == "today":
            # 오늘 평가금액
            today_info, _ = self.getCurrentScope()

            return today_info[-1]['Close'] * self.stock_amount
        elif flag == "tomorrow":
            # 내일 평가금액
            _, tomorrow_info = self.getCurrentScope()

            return tomorrow_info['Close'] * self.stock_amount
    
    def total_value(self, flag):
        # 평가금액 + 잔액
        return self.balance + self.stock_value(flag)

    def selectRange(self):
        # 오늘 이전 window_size일부터, 
        # episode_length일 후까지의 데이터 반환
        dataset = self.dataset

        row_count = len(dataset.index)
        start_index = random.randint(0, row_count - self.episode_length - self.window_size - 1)

        data = dataset.iloc[start_index:(start_index + self.episode_length + self.window_size + 1)]

        return data.to_dict('records')

    def getCurrentScope(self):
        # window_size일 전부터 오늘까지의 데이터와
        # 내일 데이터를 반환
        until_today = self.data[self.step_count:(self.step_count+self.window_size)]
        
        tomorrow = self.data[self.step_count+self.window_size]

        return until_today, tomorrow

    def __str__(self):
        # 현재 상태를 Print
        return "Step: %s, Total Value: %s, Stock Amount: %s, Balance: %s" % (self.step_count, self.total_value("tomorrow"), self.stock_amount, self.balance)

    def randomTest(self, count):
        # Random한 Action으로 테스트
        mean_value = 0
        for i in range(count):
            self.reset()

            done = False

            while not done:
                action = random.uniform(0, 1)
                obs, reward, done, info = self.step([action])
                if done == True:
                    mean_value += info['total_value']

        print("Mean value: ", mean_value / count)

if __name__ == "__main__":
    env = Stock(test=True, verbose=False)
    env.randomTest(100)
