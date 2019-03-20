import random
import numpy as np

class QAgentTemplate:
    def __init__(self, memorySize = 8192):
        self.gamma = 0.8
        self.memory = [[],[],[],[]] # s, s_, a, r
        self.memorySize = memorySize 

    def memorise(self, s, s_, a, r):
        if len(self.memory[0]) > self.memorySize:
            for i in range(4):
                self.memory[i].pop(0)

        self.memory[0].append(s)
        self.memory[1].append(s_)
        self.memory[2].append(a)
        self.memory[3].append(r)

    def getTargetQ(self):
        S = np.array(self.memory[0])
        S_ = np.array(self.memory[1])
        A = np.array(self.memory[2])
        R = np.array(self.memory[3])

        MaxQS_ = np.amax(self.getQ(S_), axis = 1) # max Q value of next states

        targetQ = self.getQ(S)
        for i in range(len(A)):
            targetQ[i,A[i]] = R[i] + self.gamma * MaxQS_[i]

        return targetQ

    def action(self, s):
        if random.uniform(0.,1.) < 0.05:
            return random.randint(0,3)
        else:
            q = self.getQ(np.array([s]))[0]
            return np.argmax(q)

    def learn(self):
        S = np.array(self.memory[0])
        targetQ = self.getTargetQ()
        self.improveQ(S, targetQ)


    # to be defined in subclass
    def getQ(self,S):
        return None

    # to be defined in subclass
    def improveQ(self, S, targetQ):
        return None