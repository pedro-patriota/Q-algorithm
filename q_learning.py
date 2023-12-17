import numpy as np
from typing import Optional
from random import uniform


class QLearning:
    """Q-Learning algorithm"""
    q_matrix = np.zeros((25, 4))
    rw = np.array(25*[0])
    state = 0

    def __init__(self, alfa:float, gama:float, epsilon:float, state: Optional[int]):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.state = state
    
    @staticmethod
    def build_state (state: str) -> tuple:
        """Build state from string"""
        plataform = int(state[:5], 2)
        direction = int(state[5:], 2)

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
    
