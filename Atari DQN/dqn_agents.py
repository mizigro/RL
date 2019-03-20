from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD

import matplotlib.pyplot as plt

from q_agent import QAgentTemplate

class DQNAgent(QAgentTemplate):
    def __init__(self, Q = None):
        super().__init__()
        self.Q = Q

    def save(self, path):
        self.Q.save(path)

    def load(self, path):
        self.Q = load_model(path)

    # S : Batch Size x State Dimensions
    def getQ(self, S):
        return self.Q.predict(S)

    # S : Batch Size x State Dimensions
    # targetQ : Batch Size x Q values
    def improveQ(self, S, targetQ):
        self.Q.fit(S, targetQ, epochs = 10, batch_size = 32, verbose = 1)



class DQNAgentCNN(DQNAgent):
    def __init__(self, in_shape, out_dim):
        Q = Sequential()

        Q.add(
            Conv2D(
                filters = 16, 
                kernel_size = 5, 
                strides = 3,
                padding = 'same', 
                activation = 'relu', 
                input_shape = in_shape
            )
        )
        
        Q.add(Flatten())
        Q.add(Dense(units = out_dim))

        Q.compile(loss='mse',optimizer = 'adam', metrics = ['accuracy'])

        super().__init__(Q)