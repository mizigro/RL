import keras
import gym
import numpy as np

class testAgent():
    def __init__(self, modelPath):
        self.Q = keras.models.load_model(modelPath)
    
    def chooseAction(self, s):
        q = self.Q.predict(np.array([s]))[0]
        return np.argmax(q)

if __name__=="__main__":
    env = gym.make('CartPole-v1')
    agent = testAgent(modelPath = 'Trained Models/450.h5')

    episodes = 10000

    for episode in range(episodes):
        r_ = 0
        s = env.reset()

        while True:
            env.render()
            a = agent.chooseAction(s)
            s_, r, done, _ = env.step(a)

            r_ += r

            if done: 
                print("Episode :",episode,"Reward :",r_)
                break

            s = s_
