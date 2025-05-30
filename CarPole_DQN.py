import gymnasium as gym
import numpy as np
import argparse
import time

class main():
    def __init__(self , args):
        # Pring hyperparameters 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")
        
        # create FrozenLake environment
        self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
        
        # Hyperparameters
        self.min_epsilon = args.min_epsilon
        self.decay_rate = args.decay_rate
        self.epochs = args.epochs
        self.alpha = args.alpha     # learning rate
        self.gamma = args.gamma     # discount factor
        self.epsilon = 1.0          # initial epsilon
        
        # Environment dimensions
        self.num_state = 4 * 4      # map size
        self.num_action = 4         # action shape :  Move [left down right up]
        
        # Initialize Q-table with zeros
        self.Q_table = np.zeros([self.num_state , self.num_action])
        
        self.train()            
        self.evaluate()
            
    def choose_action(self , state):
        # Choose an action using epsilon-greedy strategy
        random_number = np.random.random() # random float 0 ~ 1
        if random_number > self.epsilon:
            # choose the action that have max q value by current state
            action = np.argmax(self.Q_table[state]) 
        else :
            action = self.env.action_space.sample()
        return action
    
    def evaluate_action(self , state):
        # Choose the best action from Q-table (greedy policy)
        action = np.argmax(self.Q_table[state]) 
        return action
    
    def epsilon_decay(self , epoch):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * (1 - epoch / self.epochs)
        
    def update_q_table(self , state , action , next_state , reward):
        current_q_value = self.Q_table[state , action]
        max_next_q_value = np.max(self.Q_table[next_state , :])
        TD_error = reward + self.gamma * max_next_q_value
        self.Q_table[state , action] = current_q_value + self.alpha * (TD_error - current_q_value)
        
    def train(self):
        for epoch in range(self.epochs):
            print(self.epsilon)
            # reset environment
            state, info = self.env.reset()
            done = False
            while not done:
                
                # Choose action based on epsilon-greedy policy
                action = self.choose_action(state)
                
                # interact with environment
                next_state , reward , terminated, truncated, _ = self.env.step(action)     
                done = terminated or truncated
                
                # Update Q-table
                self.update_q_table(state , action , next_state , reward)
                state = next_state
                
            # Decay epsilon
            self.epsilon_decay(epoch)
            
    def evaluate(self):
        render_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

        for i in range(1000):
            # reset environment
            state, info = render_env.reset()
            done = False
            while not done:                
                action = self.evaluate_action(state)
                
                # interact with environment
                next_state , reward , terminated, truncated, _ = render_env.step(action)
                if reward == 1:
                    print("You got the treasure!!")    
                    
                done = terminated or truncated
                state = next_state
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for Q-learning")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimun epsilon")
    parser.add_argument("--decay_rate", type=float, default=0.001, help="Decay rate of epsilon")
    parser.add_argument("--epochs", type=int, default=40000, help="Maximun number of rollout steps")
    parser.add_argument("--alpha", type=float, default=0.8, help="Learning rate of q-table")
    parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate of q-table")

    args = parser.parse_args()
    
    main(args)
