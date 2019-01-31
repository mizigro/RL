import simpleEnv
import time
import numpy as np 
import random

class Agent:
    def __init__(self, env):
        self.alpha = 0.6
        self.gamma = 0.9

        self.Q = np.zeros([env.n_states+1, env.n_actions])

    def update(self, s, s_, r, a):
        # print(r)
        self.Q[s,a] = (1-self.alpha)*self.Q[s,a] + self.alpha*(r+np.max(self.Q[s_]))
    
    def action(self, s):
        return np.argmax(self.Q[s])

if __name__=="__main__":
    env = simpleEnv.Simple()
    agent = Agent(env)

    epsilon = 0.1

    for ep in range(1000):
        print("\n\nEpisode : ",ep, '\nPolicy', agent.Q)

        s = env.reset()
        env.render()

        while True:
            if random.uniform(0., 1.) < epsilon or agent.Q[s,0] == agent.Q[s,1]:
                action = random.randint(0,1)
            else:
                action = agent.action(s)

            s_, r, done = env.step(action)
            env.render()

            agent.update(s, s_, r, action)
            s = s_

            if done:
                break 

            # time.sleep(0.1)
        