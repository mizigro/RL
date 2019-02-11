import gym
import time
import random
import matplotlib.pyplot as plt
from agent import Agent

env = gym.make('CartPole-v1')

agent = Agent(env)
eps = 10000
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


plt.xlabel('Iterations')
plt.ylabel('Time Steps Lasted')
plt.plot([i for i in range(5000,10000,20)],rs[5000::20],linewidth = 0.5)
plt.savefig('performance.jpg')
env.close()