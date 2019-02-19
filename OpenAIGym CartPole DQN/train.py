import gym 
import random
import matplotlib.pyplot as plt
from dqn import DQN

if __name__=="__main__":
    env = gym.make('CartPole-v1')
    # print(env.observation_space, env.action_space)
    agent = DQN(4, 2)

    episodes = 10000
    stopLimit = 300
    Avg = []
    R = []

    for episode in range(episodes):
        r_ = 0
        s = env.reset()

        while True:
            a = agent.chooseAction(s)
            s_, r, done, _ = env.step(a)

            r_ += r

            if done: 
                agent.storeTransition(s, s_, a, -10)
                print("Episode :",episode,"Reward :",r_)
                R.append(r_)
                break

            if random.uniform(0.,1.) < 0.05:   
                agent.storeTransition(s,s_,a,r)

            s = s_

        if episode % 100 == 99:
            Avg.append(sum(R)/len(R))
            print("Average : ",Avg[-1])

            plt.plot(Avg)
            plt.savefig('performance graphs/progress.jpg')
            plt.close()
            R = []

            agent.saveModel()
            if Avg[-1] > stopLimit:
                break

            agent.learn()