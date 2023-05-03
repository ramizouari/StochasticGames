import numpy as np
from . import environment


class RLearningAgent:
    def __init__(self, env: environment.MPGEnvironment, alpha=0.1,beta=0.1, epsilon=0.1, seed=None):
        self.env: environment.MPGEnvironment = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = {}
        self.adj_table={}
        self.action_index={}
        self.beta = beta
        self.rho = 0
        if seed is None:
            seed = np.random.randint(0, 100000)
        self.rng = np.random.Generator(np.random.MT19937(seed=seed))
        for u in self.env.graph:
            self.adj_table[u] = np.array(list(self.env.graph.succ[u]))
            self.action_index[u] = {v: i for i, v in enumerate(self.adj_table[u])}
            self.q_table[u] = np.zeros(self.adj_table[u].shape)
    def get_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.adj_table[state])
        else:
            return self.adj_table[state][np.argmax(self.q_table[state])]

    def update_table(self, state, action, reward, next_state):
        max_q = np.max(self.q_table[next_state])
        action_index=self.action_index[state][action]
        dQ= reward +  max_q - self.q_table[state][action_index] - self.rho
        self.q_table[state][action_index] = self.q_table[state][action_index] + self.alpha * dQ
        if max_q == self.q_table[next_state][action_index]:
            self.rho += self.beta * (reward - self.rho + max_q - self.q_table[next_state][action_index])

    def train(self, num_episodes=1000, max_turns:int=None):
        if max_turns is None:
            max_turns = self.env.max_turns
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.get_action(int(state.observation))
            for k in range(max_turns):
                current_vertex = int(state.observation)
                next_state=self.env.step(action)
                reward = next_state.reward
                next_vertex = int(next_state.observation)
                next_action = self.get_action(next_vertex)
                self.update_table(current_vertex, action, reward, next_vertex)
                state = next_state
                action = next_action

    def play(self, env, max_turns:int=None):
        if max_turns is None:
            max_turns = self.env.max_turns
        state = env.reset()
        action = self.get_action(int(state.observation))
        mean_payoff = 0
        for k in range(max_turns):
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_action = self.get_action(next_state)
            state = next_state
            action = next_action
            mean_payoff += reward
        return mean_payoff/max_turns