import numpy as np
from typing import Optional
from random import uniform


class Functions:
    """Q-Learning algorithm"""
    q_matrix = np.zeros((25, 4))
    state = 0
    def __init__(self, alfa:float, gama:float, epsilon:float, epochs: Optional[int] = 10):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.epochs = epochs
        
    def reset(self):
        """Reset Q-matrix"""
        self.q_matrix = np.zeros((25, 4))
        self.rw = np.array(25*[0])
        self.plataform = 0
    
    @staticmethod
    def build_state (state: str) -> tuple[int, int]:
        """Build state from string"""
        
        print(state)
        state = state[2:]
        plataform = int(state[:5], 2)
        direction = int(state[5:], 2)

        print(f"{state} => plataform: {plataform}, direction: {direction}")
        return plataform, direction
    
    def epsilon_greedy_policy(self) -> int:
        """Epsilon-greedy policy"""
        random_int = uniform(0, 1)

        if random_int < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return int(np.argmax(self.q_matrix[self.state]))
    
    def greedy_policy(self) -> int:
        """Greedy policy"""
        return int(np.argmax(self.q_matrix[self.state]))
    
    def updateQMatrix(self, reward:int, next_state:str, action:int):
        plataform, _ = self.build_state(next_state)
        max_next_state =  max(self.q_matrix[plataform])
        self.q_matrix[self.state][action] = self.q_matrix[self.state][action] + self.alfa * (reward + self.gama *max_next_state - self.q_matrix[self.state][action])
