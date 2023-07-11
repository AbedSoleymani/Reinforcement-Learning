import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pygame

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self,
                 n_states=4,
                 n_hidden=16,
                 n_actions=2):
        super(PolicyNet, self).__init__()
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x
    
    """act function recieves the current state and feed it to the policy network.
    It returns the selected action as an item and the log probability of the selected action.
    The log probability can be useful for the policy gradient updates."""
    def act(self, state):
        """This line first converts the state input from a NumPy array into a PyTorch tensor.
        It is then converted to float() data type.
        Then, unsqueeze(0) adds an extra dimension to make it a 2D tensor with a batch size of 1.
        Finally, .to(device) moves the tensor to the specified device (e.g., GPU) if available."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        """.cpu() is used to move the tensor back to the CPU device."""
        probs = self.forward(state).cpu()
        """This line creates a Categorical distribution using the probabilities probs"""
        categorical_distribution = Categorical(probs)
        action = categorical_distribution.sample()
        return action.item(), categorical_distribution.log_prob(action)
    
    def REINFORCE(self,
                  env,
                  optimizer,
                  max_t=200,
                  n_episodes=1000,
                  gamma=1.0,
                  print_every=100):
        
        scores_deque = deque(maxlen=print_every)
        scores = [] # for plotting the learning trend
        for episode in range(1, n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = env.reset()[0]
            done = False

            for t in range(max_t):
                action, log_prob = self.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break

            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])

            policy_loss = []
            for log_prob in saved_log_probs:
                """log_prob at each step will be multiplied by total discounted reward in the whole episode.
                Yes, it does not make sense! But it is vanila REINFORCE algorithm!
                This also leads the algorithm to an unstable learning procedure.
                This issue is resolved in the PPO modification by replacing total discounted reward
                by the total FUTURE discounted reward which makes more sense."""
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_deque)))
                break
        
        return scores
    
    def plot(self, scores):
        
        plt.plot(np.arange(1,len(scores)+1), scores)
        plt.xlabel('Number of episodes')
        plt.ylabel('Score')
        plt.show()

    def validate(self,
                 env,
                 epochs=3,
                 max_t=500):
        
        for episode in range(1, epochs+1):
            state = env.reset()[0]
            score = 0
            for _ in range(max_t):
                env.render()
                pygame.display.set_caption('Episode {}, score: {}'.format(episode, score))
                action, _ = self.act(state)
                state, reward, done, _, _ = env.step(action)
                score += reward
                if done:
                    break