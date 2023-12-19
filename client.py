#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
from functions import Functions

s = cn.connect(2037)
action_map = {0: "north", 1: "east", 2: "south", 3: "west"}

def turn_player(action: int, direction: int):
    """Decode action"""
    
    # Define the mapping of actions to directions

    # Calculate the difference between the current direction and the desired action
    diff = (action - direction) % 4

    # Determine the appropriate action based on the difference
    if diff == 0:
        return 
    elif diff == 1:
        return [cn.get_state_reward(s, "right")]
    elif diff == 2:
        return [cn.get_state_reward(s, "right"), cn.get_state_reward(s, "right")]
    elif diff == 3:
        return [cn.get_state_reward(s, "left")]

epochs = 50
q_learning = Functions(alfa=0.8, gama=0.7, epsilon=0.1)

while True:
    state = q_learning.state

    if state == '0b0000000':
        epochs -= 1

    if epochs < 0:
        break

    platform, direction = q_learning.build_state(state=state)
    
    action = q_learning.epsilon_greedy_policy(platform)
    print(action)
    turn_player(action, direction)
    
    next_state, reward = cn.get_state_reward(s, "jump")

    q_learning.update_table(state=platform, action=action, reward=int(reward), next_state=next_state)

    
q_learning.save()