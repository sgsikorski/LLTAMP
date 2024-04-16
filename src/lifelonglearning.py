from util import utilConstants, Action, State
import torch
from GradientEpisodicMemory.model.gem import Net
import random
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, n_actions, dtype=torch.float64)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def Q(self, state, action):
        return self.forward(state)[action]

    def maxQ(self, state):
        return torch.argmax(self.forward(state))

class NetInputs():
    def __init__(self, n_layers, n_hiddens, lr, cuda, n_memories, memory_strength):
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.lr = lr
        self.cuda = cuda
        self.n_memories = n_memories
        self.memory_strength = memory_strength
        self.data_file = None

# pi_theta policy model
class Model(Net):
    def __init__(self, input_dim=3, n_tasks=30):
        args = NetInputs(n_layers=2, 
                         n_hiddens=100, 
                         lr=0.001, 
                         cuda=False, 
                         n_memories=300, 
                         memory_strength=0.5)
        # Update input_dim to map from state to features
        super(Model, self).__init__(n_inputs=input_dim, 
                                    n_outputs=len(utilConstants.ALL_ACTIONS), 
                                    n_tasks=n_tasks, 
                                    args=args)
    
    def getThetaParameter(self):
        return self.parameters()

    def predict(self, state):
        with torch.no_grad():
            return self.net(state)
    
    def update_theta(self, Buffer, t):
        # Size of s needs to equals the input_dim of the model
        s = []
        a = []
        for transition in Buffer:
            s.append([transition.state.agentX, transition.state.agentY, transition.state.agentZ])
            a.append(utilConstants.ALL_ACTIONS.index(transition.action.actionType))
        for i in range(300 - len(s)):
            s.append([0, 0, 0])
            a.append(0)
        self.observe(torch.tensor(s), t, torch.tensor(a))

    def saveModel(self, path='models/trained_agent.pt'):
        torch.save(self.net.state_dict(), path)
    
    def loadModel(self, path='models/trained_agent.pt'):
        self.net.load_state_dict(torch.load(path))