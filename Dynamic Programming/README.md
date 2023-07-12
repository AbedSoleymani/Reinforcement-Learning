Here, we will apply some classic DP methods to implement several RL concepts including Iterative Policy Evaluation, obtaining $q_\pi$ from $v_\pi$, Policy Improvement, Policy Iteration, Truncated Policy Iteration, and Value Iteration.

You can find the functions required for this concepts in `Agent` class in `agent.py` file.
#### Preparing FrozenLakeEnv for DP
Unlike the RL agent that has no idea about the dynamics of the environment, DP assumes that the agent has full knowledge of the MDP.  
To do so, we use class `MDP` in `mdp.py` file to extract the MDP process within `env = gym.make('FrozenLake-v1', is_slippery=True)`.
We would have this information  
`[(0.3333333333333333, 1, 0.0, False)`,  
 `(0.3333333333333333, 0, 0.0, False)`,  
 `(0.3333333333333333, 5, 0.0, True)]`  
if we run command 
`mdp.P[1][0]`
for the `MDP` class instance  
`mdp = MDP(num_states=env.observation_space.n, num_actions=env.action_space.n, dynamics_fn=env.unwrapped.P)`  


For the theoretical aspects of the presented codes please review the pdf file.
