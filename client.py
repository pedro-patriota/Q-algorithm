#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
from functions import Functions

s = cn.connect(2037)
action_map = {0: "north", 1: "east", 2: "south", 3: "west"}

def decodeAction(action: int, direction: int):
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

epochs = 1
q_learning = Functions(alfa=0.8, gama=0.7, epsilon=0.1)
q_learning.load()
counter = 0
for episodes in range(epochs):
    q_learning.reset()
    state = '0b0000000'
    
    while (True):
        plataform, direction = q_learning.build_state(state=state)
        q_learning.plaform = plataform
        
        counter += 1
        if counter == 1000: break
        if plataform == 24 or plataform == 25: break
    
        action = q_learning.epsilon_greedy_policy()
        decodeAction(action, direction)
        
        next_state, reward = cn.get_state_reward(s, "jump")
        print(f"{state} => plataform: {plataform}, direction: {direction}, action: {action}, nexstate={next_state}, reward={reward}")
        q_learning.updateQMatrix(action=action, reward=int(reward), next_state_total=next_state)
        state = next_state
    q_learning.save()
    
# print the final Q-matrix
print(q_learning.q_matrix)
