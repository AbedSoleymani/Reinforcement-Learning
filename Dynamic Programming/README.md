Here, we will apply some classic DP methods to implement several RL concepts including Iterative Policy Evaluation, obtaining $q_\pi$ from $v_\pi$, Policy Improvement, Policy Iteration, Truncated Policy Iteration, and Value Iteration.

You can find the functions required for this concept in `funcs.py` file.
# 0. Preparing FrozenLakeEnv for DP#
Unlike the RL agent that has no idea about the dynamics of the environment, DP assumes that the agent has full knowledge of the MDP. To do so, we add this line \n
`self.P = P`
to the original `frozenlake.py` file downloaded from OpenAI to make the one-step dynamics accessible to the agent.
# 1. Iterative Policy Evaluation#
sd
