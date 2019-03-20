import gym
from dqn_agents import DQNAgent
import time
import random
import cv2

if __name__=='__main__':
    im_w = 40
    im_h = 30

    env = gym.make('Breakout-v0')
    agent = DQNAgent()
    agent.load('models/001.h5')
    
    episodes = 1000
    
    for episode in range(episodes):
        s = env.reset()
        s, r, done, info = env.step(1)
        s = cv2.resize(s, (im_h, im_w))
        
        lives = 5
        r_ = 0

        while True:
            time.sleep(0.1)
            env.render()
            a = agent.action(s)
            s_, r, done, info = env.step(a)

            s_ = cv2.resize(s_, (im_h, im_w))

            if info['ale.lives'] < lives:
                lives = info['ale.lives']
                s, r, done, info = env.step(1)

            r_ += r

            if done:
                print('Episode:',episode,'Reward:',r_)
                break  

            s = s_
        
    
    env.close()