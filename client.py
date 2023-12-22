import connection as cn
import numpy as np
from random import uniform

def main():
    s = cn.connect(2037)

    def action(index: int) -> tuple: # type: ignore
        if index == 0:
            return cn.get_state_reward(s, "left")
        elif index == 1:
            return cn.get_state_reward(s, "right")
        elif index == 2:
            return cn.get_state_reward(s, "jump")
        
    epochs = 99999
    q_learning = QLearning(alfa=0.65, gama=0.958, epsilon=0)

    while True:
        if q_learning.state == q_learning.initial_state:
            epochs -= 1

            if epochs % 10000 == 0:
                q_learning.save(f'results/resultado{epochs / 1000}.txt') # Salvamos várias q-tables ao longo do treinamento para pegar a melhor depois
                print(epochs)

        if epochs < 0:
            break
        
        action_ind = q_learning.epsilon_greedy_policy()

        next_state, reward = action(action_ind)

        q_learning.update_table(action=action_ind, reward=int(reward), next_state=q_learning.get_state(next_state))
        #q_learning.state = q_learning.get_state(next_state)

    q_learning.save()

class QLearning:
    """Q-Learning algorithm"""
    def __init__(self, alfa:float, gama:float, epsilon:float):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.initial_state = 0*4
        self.state = self.initial_state

        self.load()

    def load(self, path='resultado.txt'):
        """Load .txt table onto an array"""
        with open(path, 'r') as file:
            lines = file.readlines()

        self.q_table = np.array([list(map(float, line.split())) for line in lines])

    def save(self, path='resultado.txt'):
        """Save q-table array to .txt file"""
        array_str = '\n'.join([' '.join(map(str, row)) for row in self.q_table])

        with open(path, 'w') as file:
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

        return platform * 4 + direction
    
main()