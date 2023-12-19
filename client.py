#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
from functions import Functions

s = cn.connect(2037)

def action(index: int) -> tuple: # type: ignore
    if index == 0:
        return cn.get_state_reward(s, "left")
    elif index == 1:
        return cn.get_state_reward(s, "right")
    elif index == 2:
        return cn.get_state_reward(s, "jump")
    
epochs = 1000
q_learning = Functions(alfa=0.1, gama=0.958, epsilon=0.9)

while True:
    if q_learning.state == 0:
        epochs -= 1

    if epochs < 0:
        break

    action_ind = q_learning.epsilon_greedy_policy()

    next_state, reward = action(action_ind)

    q_learning.update_table(action=action_ind, reward=int(reward), next_state=q_learning.get_state(next_state))

    
q_learning.save()