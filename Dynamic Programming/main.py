import gym
from mdp import MDP
from agent import *
from plot_utils import plot_values

env = gym.make('FrozenLake-v1',
               is_slippery=True)

mdp = MDP(num_states=env.observation_space.n,
          num_actions=env.action_space.n,
          dynamics_fn=env.unwrapped.P)

# print(mdp.P[1][0])

agent = Agent()

V = agent.iterative_policy_evaluation(mdp=mdp,
                                      nS=env.observation_space.n,
                                      nA=env.action_space.n,
                                      policy='random')

plot_values(V)