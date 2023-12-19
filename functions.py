import numpy as np
from typing import Optional
from random import uniform


class Functions:
    """Q-Learning algorithm"""
    q_matrix = np.zeros((25, 4))
    plaform = 0
    
    def load(self):
        with open('resultado.txt', 'r') as file:
            lines = file.readlines()

        self.q_matrix = np.array([list(map(float, line.split())) for line in lines])
    
    def __init__(self, alfa:float, gama:float, epsilon:float, epochs: Optional[int] = 10):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.epochs = epochs
    
    def save(self):
        array_str = '\n'.join([' '.join(map(str, row)) for row in self.q_matrix])

        with open('resultado.txt', 'w') as file:
            file.write(array_str)
    
    def reset(self):
        """Reset Q-matrix"""
        self.plataform = 0
    
    @staticmethod
    def build_state (state: str) -> tuple[int, int]:
        """Build state from string"""
        
        state = state[2:]
        plataform = int(state[:5], 2)
        direction = int(state[5:], 2)
        
        return plataform, direction
    
    def epsilon_greedy_policy(self) -> int:
        """Epsilon-greedy policy"""
        random_int = uniform(0, 1)

        if random_int < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return int(np.argmax(self.q_matrix[self.plaform]))
    
    def greedy_policy(self) -> int:
        """Greedy policy"""
        return int(np.argmax(self.q_matrix[self.plaform]))
    
    def updateQMatrix(self, reward:int, next_state_total:str, action:int):
        next_platform, _ = self.build_state(next_state_total)
        # max_next_state =  max(self.q_matrix[plataform])
        self.q_matrix[self.plaform][action] = (1 - self.alfa) * self.q_matrix[self.plaform][action] + self.alfa * (reward + self.gama * self.q_matrix[next_platform][action])
        # self.q_matrix[self.state][action] = self.q_matrix[self.state][action] + self.alfa * (reward + self.gama *max_next_state - self.q_matrix[self.state][action])
