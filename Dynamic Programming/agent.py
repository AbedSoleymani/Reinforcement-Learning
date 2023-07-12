import numpy as np

class Agent():
    def __init__(self) -> None:
        pass

    def iterative_policy_evaluation(self,
                          mdp,
                          nS,
                          nA,
                          policy='random',
                          gamma=1,
                          theta=1e-8):
        
        if policy == 'random':
            random_policy = np.ones([nS, nA]) / nA
            policy = random_policy
        V = np.zeros(nS)
        while True:
            delta = 0
            for s in range(nS):
                Vs = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in mdp.P[s][a]:
                        Vs += action_prob * prob * (reward + gamma * V[next_state])
                delta = max(delta, np.abs(V[s]-Vs))
                V[s] = Vs
            if delta < theta:
                break
        return V