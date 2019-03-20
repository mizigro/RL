import gym
from dqn_agents import DQNAgent
import time
import random
import cv2

if __name__=='__main__':
    im_w = 40
    im_h = 30

    env = gym.make('Breakout-v0')
    try:
        agent = DQNAgent()
        agent.load('models/001.h5')
    except Exception as e:
        print('\n',e,'\n')
        agent = DQNAgentCNN((im_w,im_h, 3), 4)
    
    episodes = 10000
    
    for episode in range(episodes):
        s = env.reset()
        s, r, done, info = env.step(1)
        s = cv2.resize(s, (im_h, im_w))


        lives = 5
        r_ = 0

        while True:
            env.render()
            # time.sleep(0.005)
            
            a = agent.action(s)
            s_, r, done, info = env.step(a)

            s_ = cv2.resize(s_, (im_h, im_w))

            if info['ale.lives'] < lives:
                agent.memorise(s,s_,a,-10)
                lives = info['ale.lives']
                s, r, done, info = env.step(1)
            else:
                if r>0:
                    agent.memorise(s,s_,a,r)
                elif random.uniform(0.,1.) > 0.05:
                    agent.memorise(s,s_,a,r)

            r_ += r

            if done:
                print('Episode:',episode,'Reward:',r_)
                break  

            s = s_
        
        if episode % 20 == 19:
            agent.learn()
            agent.save('models/001.h5')
    
    env.close()