#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
from q_learning import QLearning

s = cn.connect(2037)

def decodeAction(action: int, direction: int):
    """Decode action"""
    
    # Define the mapping of actions to directions
    action_map = {0: "north", 1: "east", 2: "south", 3: "west"}

    # Calculate the difference between the current direction and the desired action
    diff = (action - direction) % 4

    # Determine the appropriate action based on the difference
    if diff == 0:
        return []
    elif diff == 1:
        return [cn.get_state_reward(s, "right")]
    elif diff == 2:
        return [cn.get_state_reward(s, "right"), cn.get_state_reward(s, "right")]
    elif diff == 3:
        return [cn.get_state_reward(s, "left")]

epochs = 10

q_learning = QLearning(alfa=0.1, gama=0.9, epsilon=0.1)

for episodes in range(epochs):
    state, reward = cn.get_state_reward(s, "jump")

    plataform, direction = q_learning.build_state(state=state)
    q_learning.state = plataform
    
    action = q_learning.epsilon_greedy_policy()
    
    decodeAction(action, direction)
