from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

class Memory:
    def __init__(self, memory_size, batch_size):
        memory_size = int(memory_size)
        self.memory_size = memory_size
        self.batch_size = int(batch_size)
        self.counter = 0
        self.current_size = 0
        self.memory_buffer = deque(maxlen=memory_size)
    
    def store_trasition(self, s, a, r, s_):
        transition = (s, a, r, s_)
        self.memory_buffer.append(transition)
        self.counter += 1
        self.current_size = min(self.counter, self.memory_size)
    
    def uniform_sample(self):
        temp_buffer = []
        idx = np.random.randint(0, self.current_size, self.batch_size)
        for i in idx:
            temp_buffer.append(self.memory_buffer[i])
        return temp_buffer

class DQN_net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DQN_net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, a_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class DQN_model:
    def __init__(self, args, s_dim, a_dim, target_update_freq=500):
        self.args = args
        self.gamma = 0.9
        self.action_dim = a_dim
        self.dqn = DQN_net(s_dim=s_dim, a_dim=a_dim)
        lr = args.lr_DQN
        lr_base = args.base_lrs
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters())
        self.dqn_target = DQN_net(s_dim=s_dim, a_dim=a_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        step_size = int(args.max_episodes/10)
        self.scheduler_lr = torch.optim.lr_scheduler.CyclicLR(self.dqn_optimizer,
                                                                base_lr=lr,
                                                                max_lr=lr_base, step_size_up=step_size,
                                                                mode="triangular2", cycle_momentum=False)
        
        # reset number of update
        self.num_updates = 0
        self.target_update_freq = target_update_freq
        self.loss = 0.0
    
    def train(self, minibatch):
        # obtain minibatch
        state = []
        action = []
        reward = []
        state_next = []
        for tt in minibatch:
            state.append(tt[0])
            action.append(tt[1])
            reward.append(tt[2])
            state_next.append(tt[3])
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        state_next = np.array(state_next)
        state = Variable(torch.from_numpy(state)).type(dtype=torch.float32)
        action = Variable(torch.from_numpy(action)).type(dtype=torch.int64)
        reward = Variable(torch.from_numpy(reward)).type(dtype=torch.float32)
        state_next = Variable(torch.from_numpy(state_next)).type(dtype=torch.float32)
        
        Q_value = self.dqn(state)
        action = torch.unsqueeze(action, dim=1)
        Q_value = Q_value.gather(1, action)  # Q(s, a)
        Q_next = self.dqn_target(state_next).detach()
        Q_next_max, idx = Q_next.max(1)  # greedy policy
        Q_target = reward+self.gamma*Q_next_max
        Q_target = torch.unsqueeze(Q_target, dim=1)
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(Q_value, Q_target)
        self.loss = loss.data
        
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        self.num_updates += 1
        if self.num_updates%self.target_update_freq == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
    
    def e_greedy_action(self, s, epsilon):
        s = Variable(torch.from_numpy(s)).type(dtype=torch.float32)
        Q_value = self.dqn.forward(s)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim), epsilon
        else:
            return np.argmax(Q_value), epsilon
    
    def save_model(self, save_path, save_episode):
        model_path = save_path+'/net_params'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.dqn.state_dict(), model_path +
                   '/dqn_params_ep%d.pkl'%save_episode)
            