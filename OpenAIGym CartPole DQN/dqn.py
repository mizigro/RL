from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import random
import numpy as np

class DQN:

    def __init__(self, in_dim, out_dim):
        self.setupModel(in_dim, out_dim)
        self.gamma = 0.8
        self.modelSavePath = 'Trained Models/300.h5'

        self.data = []
        self.memoryLimit = 8192

    # the main model
    def setupModel(self, in_dim, out_dim):
        self.Q = Sequential()

        self.Q.add(Dense(units = 16, activation = 'relu', input_dim = in_dim))
        self.Q.add(Dense(units = 8, activation = 'relu', input_dim = in_dim))
        self.Q.add(Dense(units = out_dim))

        sgd = SGD(lr=0.001)

        self.Q.compile(loss='mse',optimizer = 'sgd', metrics = ['accuracy'])
    

    def storeTransition(self, s, s_, a, r):
        if len(self.data) > self.memoryLimit:
            self.data.pop(0)
        self.data.append( [ s, s_, a, r ] )

    def getTargetQ(self, S, S_, A, R):
        QS_ = self.Q.predict ( S_ ) # Q values for next state
        MaxQS_ = np.amax(QS_, axis = 1) # max Q value of next state

        targetQSA = R + self.gamma * MaxQS_ # target Q for state s and action a

        targetQ = self.Q.predict ( S )
        for i in range(len(A)):
            a = A[i]
            targetQ[i][a] = targetQSA[i]
        return targetQ

    def learn(self):
        S = np.array([row[0] for row in self.data])
        S_ = np.array([row[1] for row in self.data])
        A = np.array([row[2] for row in self.data])
        R = np.array([row[3] for row in self.data])

        targetQ = self.getTargetQ(S, S_, A, R)       

        self.Q.fit(S, targetQ, epochs = 5, batch_size = 32, verbose = 1)

    def chooseAction(self, s):
        if random.uniform(0.,1.) < 0.1:
            return random.randint(0,1)
        else:
            q = self.Q.predict(np.array([s]))[0]
            return np.argmax(q)

    def saveModel(self):
        self.Q.save(self.modelSavePath)