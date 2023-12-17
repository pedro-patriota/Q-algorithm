import numpy as np


class QLearning:
    """Q-Learning algorithm"""
    alfa = None
    gama = None
    q_matrix = np.array(25*[0, 0, 0, 0])
    rw = np.array(25*[0])

    def __init__(self, alfa, gama):
        self.alfa = alfa
        self.gama = gama
    
    @staticmethod
    def build_state (state: str) -> tuple:
        """Build state from string"""
        plataform = int(state[:5], 2)
        direction = int(state[5:], 2)

        return plataform, direction


        
