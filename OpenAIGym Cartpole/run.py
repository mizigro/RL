import gym
import time
import random
import matplotlib.pyplot as plt
from agent import Agent

env = gym.make('CartPole-v1')

agent = Agent(env)
eps = 5000
tests = 10

rs = []

# print(env.observation_space.high,env.observation_space.low)
# input()

for ep in range(eps):
    obv = env.reset()
    s = agent.obvToState(obv)
    r = 0

    while True:
        if ep>eps-tests:
            env.render()
            # print(s, agent.Q[tuple(s)],np.argmax(agent.Q[tuple(s)]))
            # time.sleep(0.1)

        epsilon = 0.3*(1-ep/eps)
        if random.uniform(0.,1.) < epsilon and ep<eps-tests:
            action = env.action_space.sample()
        else:
            action = agent.action(s)

        obv, reward, done, info = env.step(action)
        s_ = agent.obvToState(obv)

        r += reward

        if done:
            agent.updatePolicy(s, s_, action, -1)
            # print("Episode :",ep, "Reward :", r)
            rs.append(r)
            if ep*10 % eps == 0:
                print('%d%% complete' % (ep/eps*100),end="\r")
            break
        else:
            agent.updatePolicy(s, s_, action, reward)
            s = s_

plt.plot(rs)
plt.show()
env.close()