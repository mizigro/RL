import numpy as np

MIN = -3.0
MAX = 3.0

class Agent:
    def __init__(self, env, lr = 0.4):
        self.env = env
        self.low = self.env.observation_space.low 
        self.high = self.env.observation_space.high

        self.divs = 7
        self.Q = np.zeros(tuple([self.divs for i in range(env.observation_space.shape[0])])+tuple([env.action_space.n]))
        self.gma = 0.6
        self.lr = lr

    def obvCvt(self, obv, i):
        high = min(self.high[i], MAX)
        low = max(self.low[i], MIN)
        val = max(min(obv[i], MAX),MIN)
        return int((val - low)/(high - low)*self.divs)-1

    def obvToState(self, obv):
        return [self.obvCvt(obv, i) for i in range(len(obv))]

    def action(self, s):
        return np.argmax(self.Q[tuple(s)])

    def updatePolicy(self, s, s_, a, r):
        prev = self.Q[tuple(s)]
        prev[a] = (1-self.lr) * prev[a] + self.lr*(r + self.gma*(np.max(self.Q[tuple(s_)])))