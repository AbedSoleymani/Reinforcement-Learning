import torch
from environment import Environment
from policy_net import PolicyNet
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Visualizing the naive agent behavior in the environment
env_obj = Environment(name='CartPole-v1', render=True)
env_obj.test_env(n_episodes=1, policy='random')


# Training the policy using the REINFORCE algorithm
env_obj = Environment(name='CartPole-v1', render=False)
env = env_obj.create()
policy = PolicyNet().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
scores = policy.REINFORCE(env=env, optimizer=optimizer, n_episodes=1000, print_every=100)
policy.plot(scores=scores)

# Visualizing the smart agent
env = Environment(name='CartPole-v1', render=True).create()
policy.validate(env=env)