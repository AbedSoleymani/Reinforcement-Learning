import gym
import pygame
import time

class Environment():
    def __init__(self, render=True, name='CartPole-v1'):
        self.name = name
        self.render = render

    def create(self):
        if self.render:
            env = gym.make(self.name, render_mode="human")
        else:
            env = gym.make(self.name)
        return env
    
    def test_env(self, n_episodes=5, policy='random'):
        env = self.create()

        for episode in range(1, n_episodes+1):
            states = env.reset()
            done = False
            score = 0
            while not done:
                env.render()
                pygame.display.set_caption('Episode {}, score: {}'.format(episode, score))
                if policy == 'random':
                    action = env.action_space.sample()
                state, reward, done, truncated, info = env.step(action)
                score += reward
                time.sleep(0.05)
            print("Episode {}, Total score={}".format(episode, score))