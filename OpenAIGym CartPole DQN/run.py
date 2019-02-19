import gym 
import random
from dqn import DQN

if __name__=="__main__":
    env = gym.make('CartPole-v1')
    # print(env.observation_space, env.action_space)
    agent = DQN(4, 2)

    episodes = 10000
    render = False
    R = []

    for episode in range(episodes):
        r_ = 0
        s = env.reset()

        while True:
            if episode > episodes - 10:
                env.render()
            a = agent.chooseAction(s)
            s_, r, done, _ = env.step(a)

            r_ += r

            if done: 
                agent.storeTransition(s, s_, a, -10)
                print("Episode :",episode,"Reward :",r_)
                if r_ > 200:
                    render = True
                else:
                    render = False
                R.append(r_)
                break

            if random.uniform(0.,1.) < 0.05:   
                agent.storeTransition(s,s_,a,r)

            s = s_

        if episode % 100 == 99:
            print("Average : ",sum(R)/len(R))
            input()
            R = []
            agent.learn()