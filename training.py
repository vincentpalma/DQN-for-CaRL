import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle

import carrl

# Env parameters

AVAILABLE_ACTIONS = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]

# Hyperparameters
EPS_START = 0.9     # Epsilon start value
EPS_END = 0.05      # Epsilon end value
EPS_DECAY = 200     # Epsilon decay
GAMMA = 0.95        # Discount factor
LR = 0.001          # Learning rate
HIDDEN_LAYER = 256  # Hidden layer size
BATCH_SIZE = 64     # Batch size

### DQN
class DQN(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self.l1 = nn.Linear(55, HIDDEN_LAYER)    # input is img vector of 55 dimensions
    self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
    self.l3 = nn.Linear(HIDDEN_LAYER,len(AVAILABLE_ACTIONS))       

  def forward(self, x):   # forward propagation
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.l3(x)
    return x

model = DQN()
optimizer = optim.Adam(model.parameters(), LR)  # Adam optimizer is used

### Action selection with Epsilon (Exploration/Exploitation)
steps_done = 0
def select_action(state):
  global steps_done     # we want to access the variable outside of the scope
  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
  steps_done += 1
  if sample > eps_threshold:
    return model(torch.FloatTensor(state)).data.max(1)[1].view(1, 1) # Feeds the NN with S, gets index of best A
  else:   
    return torch.LongTensor([[random.randrange(len(AVAILABLE_ACTIONS))]]) # return random action

### Memory Replay
class ReplayMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []

  def push(self, transition):
    self.memory.append(transition)
    if len(self.memory) > self.capacity:
      del self.memory[0]      # It is also possible to use deque from collections

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


memory = ReplayMemory(1e6)
scores = []
best_score = 0
save_checkpoint = 0

def run_episode(e, env,logs=True):
  env.reset()
  env.target_size = (320,176) # The size of the window goes to 1280x704 when env is reset...
  state = env.step([0,0])[0]
  score = 0
  while True:
    #env.render()
    action = select_action(torch.FloatTensor([state]))
    next_state, reward, done, _ = env.step(AVAILABLE_ACTIONS[action.numpy()[0, 0]])
    
    if done: # negative reward when the car goes out of the road and the ep ends
      reward = -10

    memory.push((torch.FloatTensor([state]),
                 action,  # action is already a tensor
                 torch.FloatTensor([next_state]),
                 torch.FloatTensor([reward])))

    learn()

    state = next_state
    score += reward
    global best_score      
    global save_checkpoint
    global model      # I know it's ugly..

    if done:
      print(f"Episode {e} finished with score {score}")
      scores.append(score)
      if e%10==0 and logs: # Logs saved in TrainingStats folder
        filename = "TrainingStats/log_"+ str(e) +"_scores.pkl"
        pickle.dump(scores,open(filename,"wb"))
      if score > best_score:
        best_score = score
        if e > 1: # Sometimes we don't want the weights generated at the beginning
          torch.save(model, 'SavedWeights/dqtrain/'+str(save_checkpoint)+'dqn_'+str(e)+'.pt')
          save_checkpoint += 1
      break


### Train function
def learn():
  if len(memory) < BATCH_SIZE:
    return  # can't learn yet

  # Random transition batch is taken from experience replay memory
  transitions = memory.sample(BATCH_SIZE)
  batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions) #(S,A,S1,R)

  batch_state = Variable(torch.cat(batch_state))
  batch_action = Variable(torch.cat(batch_action))
  batch_reward = Variable(torch.cat(batch_reward))
  batch_next_state = Variable(torch.cat(batch_next_state))

  # Current Q values are estimated by NN for all actions
  current_q_values = model(batch_state).gather(1, batch_action)
  # Expected Q values are estimated from actions which gives maximum Q value
  max_next_q_values = model(batch_next_state).detach().max(1)[0]
  expected_q_values = batch_reward + (GAMMA * max_next_q_values)

  # Loss is measured from error between current and newly expected Q values
  loss = F.smooth_l1_loss(current_q_values.reshape_as(expected_q_values), expected_q_values)

  # Backpropagation of loss to NN
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

### Training

EPISODES = 15000  # number of episodes
#establish the environment
env = carrl.carl()

for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
pickle.dump(scores, open("TrainingStats/finalscores.pkl","wb"))

plt.plot([i+1 for i in range(EPISODES)],scores)
plt.show()