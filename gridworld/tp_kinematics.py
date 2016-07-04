import numpy as np
import math
# AXIS INFORMATION:
			
#				.-----> x
#				|
#				|
#				|
#				| y
#
#				actions: 0 = stay
#						 1 = x +1
#						 2 = x -1
#						 3 = y +1
#						 4 = y -1


def toy_kinematics_gridworld(state,action,boundaries):
	# state is a state vector of : [robot_x,robot_y,obstacle_x,obstacle_y,obstacle_v]
	# boundaries = [gridworld_x,gridworld_y]
	#target is always static
	new_state = np.zeros(len(state))
	if state[0]<boundaries[0] and action==1:
		new_state[0] = state[0]+ 1
	elif state[0]>0 and action==2:
		new_state[0] = state[0] - 1
	else: 
		new_state[0] = state[0]


	if state[1]<boundaries[1] and action==3:
		new_state[1] = state[1]+ 1
	elif state[1]>0 and action==4:
		new_state[1] = state[1] - 1
	else: 
		new_state[1] = state[1]


	if state[2]<boundaries[0] and state[4]==1:
		new_state[2] = state[2]+ 1
	elif state[2]>0 and state[4]==2:
		new_state[2] = state[2] - 1
	elif state[2]==boundaries[0] and state[4]==1:
		new_state[2] = 0
	elif state[2]==0 and state[4]==2:
		new_state[2] = boundaries[0]
	else:
		new_state[2] = state[2]


	if state[3]<boundaries[1] and state[4]==3:
		new_state[3] = state[3] + 1
	elif state[3]>0 and state[4]==4:
		new_state[3] = state[3] - 1
	elif state[3]==boundaries[1] and state[4]==3:
		new_state[3] = 0
	elif state[3]==0 and state[4]==4:
		new_state[3] = boundaries[1]
	else:
		new_state[3] = state[3]


	new_state[4] = state[4]
	return new_state


if __name__=="__main__":
	#out = staticGroup_with_target([0.5,0.5,0.7,0.9],[0.1,0.2])
	state = [0,15,3,3,4]
	action = 0
	for i in range(25):
		state = toy_kinematics_gridworld(state,action,[15,15])
		print state