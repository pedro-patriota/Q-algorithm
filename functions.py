import numpy as np
from random import uniform


class Functions:
    """Q-Learning algorithm"""
    def __init__(self, alfa:float, gama:float, epsilon:float):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.state = 0

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

    
    def epsilon_greedy_policy(self) -> int:
        """Epsilon-greedy policy"""
        random_int = uniform(0, 1)

        if random_int < self.epsilon:
            return np.random.randint(0, 3)
        else:
            return int(np.argmax(self.q_table[self.state]))
    
    def update_table(self, reward:int, next_state:int, action:int):
        best_next_action = np.argmax(self.q_table[next_state])
        updated_q_value = (1 - self.alfa) * self.q_table[self.state, action] + self.alfa * (reward + self.gama * self.q_table[next_state, best_next_action])  # type: ignore
        self.q_table[self.state, action] = updated_q_value

        self.state = next_state

    @staticmethod
    def get_state(state: str) -> int:
        """Convert state from binary string to integer"""
        
        state = state[2:]
        platform = int(state[:5], 2)
        direction = int(state[5:], 2)

        return platform + direction