Here, we will apply some classic DP methods to implement several RL concepts including Iterative Policy Evaluation, obtaining $q_\pi$ from $v_\pi$, Policy Improvement, Policy Iteration, Truncated Policy Iteration, and Value Iteration.

You can find the functions required for this concepts in Agent class in `agent.py` file.
#### Preparing FrozenLakeEnv for DP
Unlike the RL agent that has no idea about the dynamics of the environment, DP assumes that the agent has full knowledge of the MDP.  
To do so, we add the line
`self.P = P`
to the original `frozenlake.py` file downloaded from OpenAI to make the one-step dynamics accessible to the agent.  
In this way, for the object `env = FrozenLakeEnv(is_slippery=True)` we would have this information  
`[(0.3333333333333333, 1, 0.0, False)`,  
 `(0.3333333333333333, 0, 0.0, False)`,  
 `(0.3333333333333333, 5, 0.0, True)]`  
for this command:  
`env.P[1][0]`


For the theoretical aspects of the presented codes please review the pdf file.
