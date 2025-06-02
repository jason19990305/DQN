import gymnasium as gym
import numpy as np
import argparse
import time

from DQN.Agent import Agent

class main():
    def __init__(self , args):
        
        args.num_states = 4 # position , velocity , pole angle , pole angular velocity
        args.num_actions = 2 # left or right
        # Pring hyperparameters 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")
        
        # create FrozenLake environment
        env = gym.make('CartPole-v1')#sutton_barto_reward=True
        
        self.agent = Agent(args, env , [128,128]) # hidden layer size   
        
        self.agent.train()       
        render_env = gym.make('CartPole-v1', render_mode="human")  
        for i in range(1000):
            self.agent.evaluate(render_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DQN")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum value of epsilon for exploration")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--decay_rate", type=float, default=0.995, help="Epsilon decay rate per episode")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update rate for target network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    

    args = parser.parse_args()
    
    main(args)
