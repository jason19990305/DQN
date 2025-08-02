from DQN.ReplayBuffer import ReplayBuffer
from DQN.Network import Network
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Agent():
    def __init__(self , args , env , hidden_layer_list=[64,64]):
        # Hyperparameter
        self.buffer_size = args.buffer_size
        self.min_epsilon = args.min_epsilon
        self.batch_size = args.batch_size
        self.decay_rate = args.decay_rate
        self.epochs = args.epochs
        self.gamma = args.gamma    
        self.tau = args.tau 
        self.lr = args.lr
                 
        
        # Variable
        self.episode_count = 0
        self.epsilon = 0.9 # initial epsilon

        
        # other
        self.env = env
        self.replay_buffer = ReplayBuffer(args)
        # The model interacts with the environment and gets updated continuously
        self.eval_Q_model = Network(args , hidden_layer_list.copy())
        

        # The model be replaced regularly
        self.target_Q_model = Network(args , hidden_layer_list.copy())
        
        # Copy the parameters of eval_Q_model to target_Q_model
        self.target_Q_model.load_state_dict(self.eval_Q_model.state_dict())
        self.optimizer_Q_model = torch.optim.Adam(self.eval_Q_model.parameters() , lr = self.lr , eps=1e-5)
        self.Loss_function = nn.SmoothL1Loss()
        print(self.eval_Q_model)
        print(self.target_Q_model)
        print("-----------")

    def choose_action(self, state):
        # Epsilon-greedy action selection
        with torch.no_grad():            
            random_number = np.random.random()  # Random float in [0, 1)
            if random_number > self.epsilon:
                # Exploitation: choose the action with the highest Q-value
                state = torch.unsqueeze(torch.tensor(state), dim=0)
                action = torch.argmax(self.eval_Q_model(state)).squeeze().numpy()
            else:
                # Exploration: choose a random action
                action = self.env.action_space.sample()
        return action

    def evaluate_action(self, state):
        with torch.no_grad():
            # choose the action that have max q value by current state
            state = torch.unsqueeze(torch.tensor(state), dim=0)
            action = torch.argmax(self.eval_Q_model(state)).squeeze().numpy()
        return action

    def epsilon_decay(self, epoch):
        self.epsilon = self.epsilon * self.decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
        
    def train(self):
        episode_reward_list = []
        episode_count_list = []
        episode_count = 0
        # Training loop
        for epoch in range(self.epochs):
            # reset environment
            state, info = self.env.reset()
            done = False
            while not done:
                
                # Choose action based on epsilon-greedy policy
                action = self.choose_action(state)
                
                # interact with environment
                next_state , reward , terminated, truncated, _ = self.env.step(action)   
                done = terminated or truncated
                self.replay_buffer.store(state, action, [reward], next_state, [done])
                
                state = next_state

                # Update Q-table
                if self.replay_buffer.count > self.batch_size:
                    self.update()
            # Decay epsilon
            self.epsilon_decay(epoch)            

            if epoch % 10 == 0:
                evaluate_reward = self.evaluate(self.env)
                print("Epoch : %d / %d\t Reward : %0.2f"%(epoch,self.epochs , evaluate_reward))
                episode_reward_list.append(evaluate_reward)
                episode_count_list.append(episode_count)
                
            episode_count += 1

        # Plot the training curve
        plt.plot(episode_count_list, episode_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()

    def update(self):
        #print("------------------")
        s, a, r, s_, done = self.replay_buffer.numpy_to_tensor()  # Get training data .type is tensor
        index = np.random.choice(len(r), self.batch_size, replace=False)
        minibatch_s = s[index]
        minibatch_a = a[index]
        minibatch_r = r[index]
        minibatch_s_ = s_[index]
        minibatch_done = done[index]
        minibatch_a = minibatch_a.view(-1,1)
        
        # Use target network to calculate the TD-error
        with torch.no_grad():
            next_value = self.target_Q_model(minibatch_s_).max(dim=1, keepdim=True).values
            target_value = minibatch_r + self.gamma * next_value * (1 - minibatch_done)
            
        # MSE
        current_value = self.eval_Q_model(minibatch_s).gather(dim=1, index=minibatch_a)
        loss = self.Loss_function(current_value, target_value)
        self.optimizer_Q_model.zero_grad()
        loss.backward()
        self.optimizer_Q_model.step()

        self.soft_update(self.target_Q_model, self.eval_Q_model, self.tau)

    def soft_update(self, target, eval, tau):
        for target_param, eval_param in zip(target.parameters(), eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + eval_param.data * tau)

    def evaluate(self, env):
        render_env = env

        reward_list = []
        for i in range(5):
            # reset environment
            state, info = render_env.reset()
            done = False
            episode_reward = 0
            expected_reward_list = []

            while not done:                
                action = self.evaluate_action(state)
                
                # interact with environment
                next_state , reward , terminated, truncated, _ = render_env.step(action)
                with torch.no_grad():
                    tensor_next_state = torch.unsqueeze(torch.tensor(next_state), dim=0)
                    expected_reward = reward + self.gamma * self.target_Q_model(tensor_next_state).max().item()
                    expected_reward_list.append(expected_reward)
                    
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            reward_list.append(episode_reward)
        reward_list = np.array(reward_list)
        return reward_list.mean()