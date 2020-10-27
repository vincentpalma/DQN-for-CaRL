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
import numpy as np

import carrl

# Env parameters

AVAILABLE_ACTIONS = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]

# Hyperparameters
EPS_START = 0.9     # Epsilon start value
EPS_END = 0.05      # Epsilon end value
EPS_DECAY = 200     # Epsilon decay
GAMMA = 0.99        # Discount factor
LR = 0.0003         # Learning rate
HIDDEN_LAYER = 256  # Hidden layer size
BATCH_SIZE = 64     # Batch size

### DQN
class DQN(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self.l1 = nn.Linear(59, HIDDEN_LAYER)    # input is vector of 59 dims (flatted img + 2 last actions)
    self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
    self.l3 = nn.Linear(HIDDEN_LAYER,len(AVAILABLE_ACTIONS))       

  def forward(self, x):   # forward propagation
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.l3(x)
    return x

model = DQN()
optimizer = optim.Adam(model.parameters(), LR)  # Adam optimizer is used for backpropagation

### Action selection with Epsilon (Exploration/Exploitation)
steps_done = 0
def select_action(state):
  global steps_done     # we want to access the variable outside of the scope
  sample = random.random()
  epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
  steps_done += 1
  if sample > epsilon:
    return model(torch.FloatTensor(state)).data.max(1)[1].view(1, 1) # Feeds the NN with S, gets index of best A
  else:   
    return torch.LongTensor([[random.randrange(len(AVAILABLE_ACTIONS))]]) # return random action

### Experience Replay 
class ReplayMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []

  def push(self, transition):
    self.memory.append(transition)
    if len(self.memory) > self.capacity:
      del self.memory[0]      

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
  step = 0
  env.target_size = (320,176) # The size of the window goes to 1280x704 when env is reset...
  state = env.step([0,0])[0]
  state = np.append(state,[0,0,0,0])  # Add 2 last actions
  score = 0
  last_action = select_action(torch.FloatTensor([state]))
  while True:
    #env.render()
    action = select_action(torch.FloatTensor([state]))
    next_state, reward, done, _ = env.step(AVAILABLE_ACTIONS[action.numpy()[0, 0]])
    
    next_state = np.append(next_state,np.append(AVAILABLE_ACTIONS[last_action.numpy()[0, 0]],AVAILABLE_ACTIONS[action.numpy()[0, 0]]))
    #print(f'DEBUG: {np.shape(next_state)} last Action{last_action} / Action{action}')

    last_action = action

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

    step += 1
    if step >= 4000:
      done = True # We stop after 4000 steps
    if done:
      print(f"Episode {e} finished with score {score}")
      scores.append(score)
      if e%10==0 and logs: # Logs saved in TrainingStats folder
        filename = "TrainingStats/scores4k.pkl"
        pickle.dump(scores,open(filename,"wb"))
      if score > best_score:
        best_score = score
        if e > 50: # Sometimes we don't want the weights generated at the beginning
          torch.save(model, 'SavedWeights/dqtrain_limit4k/'+str(save_checkpoint)+'dqn_'+str(e)+'.pt')
          save_checkpoint += 1
      break


### Train function
def learn():
  if len(memory) < BATCH_SIZE:
    return  # Can't learn yet because memory is not full

  # Random transition batch is taken from experience replay memory
  transitions = memory.sample(BATCH_SIZE)
  batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions) #(Sj,Aj,Sj+1,Rj+1)

  batch_state = Variable(torch.cat(batch_state))
  batch_action = Variable(torch.cat(batch_action))
  batch_reward = Variable(torch.cat(batch_reward))
  batch_next_state = Variable(torch.cat(batch_next_state))

  # Current Q values are estimated by NN for all actions
  current_q_values = model(batch_state).gather(1, batch_action)
  # Expected Q values are estimated from actions which gives maximum Q value
  max_next_q_values = model(batch_next_state).detach().max(1)[0]
  expected_q_values = batch_reward + (GAMMA * max_next_q_values) #calculating y_j

  # Loss is measured from error between current and newly expected Q values
  loss = F.smooth_l1_loss(current_q_values.reshape_as(expected_q_values), expected_q_values)

  # Backpropagation of loss to NN
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

### Training

episodes = 800
env = carrl.carl()

for episode in range(episodes):
  run_episode(episode, env)

print('Complete')
pickle.dump(scores, open("TrainingStats/scores4k.pkl","wb"))

plt.plot([i+1 for i in range(episodes)],scores)
plt.show()