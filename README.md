# Reinforcement Learning  

### Simple Q Learning Test

Simple 1D game having a starting point and an ending point separated by 5 steps. Objective : reach from one end to the other end


### Open AI Gym (Cartpole environment)

*An attempt to solve the cartpole problem simply using the bellman equation for Q Learning* - After about 1000 iterations, the algorithm is seen to last around 150 to 200 time steps on an average.


<img src="https://github.com/mizimo/ReinforcementLearning/raw/master/OpenAIGym%20Cartpole/performance.jpg" width="400px">


### DQN - Open AI Gym (Cartpole environment)

*Solving OpenAI Gym CartPole Problem using a Deep Q Network with 2 layers and ReLU activations* - The network can reach 300+ time steps on an average in a few hundred iterations. A particular trained model saved as 450.h5 in the models folder beat the game by lasting 500 time steps consecutively non stop for as many test episodes it was run. 

<img src="https://github.com/mizimo/ReinforcementLearning/raw/master/OpenAIGym%20CartPole%20DQN/performance%20graphs/progress.jpg" width="400px">
