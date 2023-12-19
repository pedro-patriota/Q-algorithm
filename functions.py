import numpy as np
from typing import Optional
from random import uniform


class Functions:
    """Q-Learning algorithm"""
    def __init__(self, alfa:float, gama:float, epsilon:float, epochs: Optional[int] = 10):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.epochs = epochs
        self.state = '0b0000000'
        self.state_int = 0

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
        plataform = int(state[:5], 2)
        direction = int(state[5:], 2)

        #print(f"{state} => plataform: {plataform}, direction: {direction}")
        return plataform, direction
    
    def epsilon_greedy_policy(self) -> int:
        """Epsilon-greedy policy"""
        random_int = uniform(0, 1)

        if random_int < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return int(np.argmax(self.q_table[self.state]))
    
    def greedy_policy(self) -> int:
        """Greedy policy"""
        return int(np.argmax(self.q_table[self.state]))
    
    def updateQMatrix(self, reward:int, next_state:str, action:int):
        plataform, _ = self.build_state(next_state)
        max_next_state =  max(self.q_table[plataform])
        self.q_table[self.state][action] = self.q_table[self.state][action] + self.alfa * (reward + self.gama * max_next_state - self.q_table[self.state][action])
