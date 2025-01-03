# My attempt to implement the experiment from the paper 
# "Playing Atari with Deep Reinforcement Learning" by Mnih et al.

# Author: Liam Gaeuman

import gymnasium as gym
from collections import deque, namedtuple
import random
import cv2
import math
import torch  
import numpy as np
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import pickle

PONG = "ALE/Pong-v5" 
BREAKOUT = "ALE/Breakout-v5"
ICE_HOCKEY = "ALE/IceHockey-v5"

class Deep_Q_Learning:
    # Algorithm 1 Deep Q-learning with Experience Replay
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.1
    EPS_DECAY = 30000
    LR = 0.00001
    criterion = nn.MSELoss()
    steps_done = 0
    BATCH_SIZE = 32
    
    def __init__(self, replay_capacity : int, environment : str, frame_skip=4):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(self.device)
        self.env = gym.make(environment)
        self.preprocessor = Preprocessor()  #see if you can delete this 
        # Initialize replay memory D to capacity N
        self.memory = ReplayMemory(replay_capacity)
        # Initialize action-value function Q with random weights
        self.cnn = SimpleCNN(self.env.action_space.n, self.device).to(self.device)
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.LR)
        self.frame_skip = frame_skip
        
    def run(self):  
        epoch_complete = False
        total_rewards = 0
        episode_count = 0
        #avg_reward_per_epoch = []

        # for episode = 1, M do
        while(self.steps_done < 10 ** 7):

            #initialize environment and get its observation
            observation, _ = self.env.reset()
            # Initialise sequence s1 = {x1} and preprocessed sequence φ1 = φ(s1)
            self.preprocessor.clear()
            states = []
            states.append(self.prep_state(self.preprocessor.push(observation)))

            if epoch_complete:
                avg = total_rewards/episode_count
                print(avg)
                #avg_reward_per_epoch.append(avg)
                total_rewards = 0
                episode_count = 0
                epoch_complete = False

            # for t = 1, T do
            i = 0
            rewards = 0
            while(1):
                # With probability ε select a random action a_t otherwise select a_t = maxa Q*(φ(s_t), a; θ)
                if i % self.frame_skip == 0:
                    action = self.select_action(states[-1])
                # Execute action at in emulator and observe reward r_t and image x_t+1
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.steps_done += 1
                rewards += reward
                # Set s_t+1 = s_t, a_t, x_t+1 and preprocess φ_t+1 = φ(s_t+1)
                states.append(self.prep_state(self.preprocessor.push(observation)))
                i += 1
                # Store transition (φ_t, a_t, r_t, φ_t+1) in D
                self.memory.push((states[-2], action, reward, states[-1], terminated))
                
                self.optimize_model()

                if self.steps_done % 50000 == 0:
                    epoch_complete = True
                
                if terminated or truncated:
                    total_rewards += rewards
                    episode_count += 1
                    break
                
        self.pickle_model()
        print("done")

    
    @staticmethod
    def prep_state(state):
        state = torch.from_numpy(state)
        state = state.permute(2, 0, 1)
        return state.float()

    def select_action(self, state):
        sample = random.random()      
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                output = self.cnn(state.unsqueeze(0).to(self.device))
            x = output.argmax(dim=1).item()
            return x

    
        return self.env.action_space.sample()
    
    
    # (state, action, reward, next state, next state terminal)
    def optimize_model(self):
        # Sample random minibatch of transitions (φ_j, a_j, r_j, φ_j+1) from D
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = self.memory.sample(self.BATCH_SIZE)
        # Set y_j = { r_j for terminal φ_j+1
        #            r_j + γ maxa' Q(φ_j+1, a'; θ) for non-terminal φ_j+1 }
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
  
        # Get current Q values from Q-network
        q_values = self.cnn(states).gather(1, actions.unsqueeze(1)).squeeze(1) #gets the values for the action that was taken 
        next_q_values = self.cnn(next_states).max(1)[0] #returns the max q values for each (the action that should have been taken)

        # Calculate target values
        targets = rewards + (self.GAMMA * next_q_values * (1 - dones)).to(self.device)

        # Compute the loss
        loss = self.criterion(q_values, targets)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pickle_model(self):
        with open('breakout_nn.pkl', 'wb') as f:
            pickle.dump(self.cnn, f)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition : tuple):
        """Save a transition"""
        # (state, action, reward, next state, next state terminal)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class SimpleCNN(nn.Module):
    def __init__(self, output_dim, device):
        super(SimpleCNN, self).__init__()
        
        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 21 * 21, 128)  # Adjust based on output size of conv layers
        self.fc2 = nn.Linear(128, output_dim)  # Set output_dim based on the task
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 21 * 21)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Preprocessor:
    HEIGHT = 84
    WIDTH = 84

    def __init__(self):
        self.deque = deque(maxlen=4)  # Max length to keep only the last 4 frames

    def push(self, frame):
        # Preprocess the incoming frame
        preprocessed_frame = self.preprocess_frame(frame)
        # Add the preprocessed frame to the deque
        self.deque.appendleft(preprocessed_frame)
        # If deque has fewer than 4 frames, pad with copies of the first frame
        while len(self.deque) < 4:
            self.deque.appendleft(preprocessed_frame)
        return self.preprocess()

    def preprocess_frame(self, frame):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize frame
        resized_frame = cv2.resize(gray_frame, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA) # make sure this is actually downsampling followed by cropping
        # Normalize pixel values to range [0, 1]
        normalized_frame = resized_frame / 255.0
        return normalized_frame

    def preprocess(self):
        # Stack frames along the last dimension to create a tensor with shape (84, 84, 4)
        stacked_frames = np.stack(self.deque, axis=-1)
        return stacked_frames
    
    def clear(self):
        self.deque.clear()


if __name__ == '__main__':

    replay_capacity = 10**6
    environment = "ALE/Breakout-v5"
    algorithm = Deep_Q_Learning(replay_capacity, environment)
    algorithm.run()