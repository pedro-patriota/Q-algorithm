import numpy as np
from typing import Optional
from random import uniform


class Functions:
    """Q-Learning algorithm"""
    def __init__(self, alfa:float, gama:float, epsilon:float):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.state = '0b0000000'

        self.load()
        self.save()

    def load(self):
        with open('resultado.txt', 'r') as file:
            lines = file.readlines()

        self.q_table = np.array([list(map(float, line.split())) for line in lines])

    def save(self):
        array_str = '\n'.join([' '.join(map(str, row)) for row in self.q_table])

        with open('resultado.txt', 'w') as file:
            file.write(array_str)

    def update_state(self, new_state):
        self.state = new_state
        self.state_int = int(self.state[2:7], 2)

    @staticmethod
    def build_state (state: str) -> tuple[int, int]:
        """Build state from string"""
        
        state = state[2:]
        platform = int(state[:5], 2)
        direction = int(state[5:], 2)

        return platform, direction
    
    def epsilon_greedy_policy(self, state) -> int:
        """Epsilon-greedy policy"""
        random_int = uniform(0, 1)

        if random_int < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return int(np.argmax(self.q_table[state]))
    
    def greedy_policy(self) -> int:
        """Greedy policy"""
        return int(np.argmax(self.q_table[self.state]))
    
    def update_table(self, state, reward:int, next_state:str, action:int):
        next_platform, _ = self.build_state(next_state)

        best_next_action = np.argmax(self.q_table[next_platform])
        updated_q_value = (1 - self.alfa) * self.q_table[state, action] + self.alfa * (reward + self.gama * self.q_table[next_platform, best_next_action]) 
        self.q_table[state, action] = updated_q_value

        self.state = next_state
        #self.q_table[self.state][action] = self.q_table[self.state][action] + self.alfa * (reward + self.gama * max_next_state - self.q_table[self.state][action])
