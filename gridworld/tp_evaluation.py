import tp_forwardBackward as fb
import numpy as np
import matplotlib.pyplot as plt

def eval_value(target_weights,apprentice_policy,model,initial_states,time_steps):
		a_state_freq,a_state_action_freq,a_all = fb.forward(apprentice_policy,model.transition,initial_states,time_steps) # 30 timesteps
		a,s,f = model.feature_f.shape
		feature_exp = np.dot(a_state_action_freq.reshape(s*a),model.feature_f.reshape(s*a,f))
		value = np.dot(feature_exp,target_weights)
		return value

def eval_reward_f(expert_reward,apprentice_reward):
	avg1 = expert_reward - np.min(expert_reward)
	avg2 = apprentice_reward - np.min(apprentice_reward)
	#print "AVERAGE EXPERT",avg1
	#print "AVERAGE Apprentice",avg2
	return np.sum(abs(avg1-avg2))

def plot_state(all_states,target,boundaries,alph = 0.1,axis = None):
	#red = np.zeros([all_states.shape[0],4]);
	alph = abs(alph-0.001)
	green = np.zeros([all_states.shape[0],4]);blue = np.zeros([all_states.shape[0],4])
	blue[:,:3] = np.array([0,0,1]*all_states.shape[0]).reshape(all_states.shape[0],3) ; blue[:,3] = list(alph)
	green[:,:3] = np.array([0,1,0]*all_states.shape[0]).reshape(all_states.shape[0],3) ; green[:,3] = list(alph)
	axis.scatter(target[0],target[1],color="r",alpha = 0.0001)
	axis.scatter(all_states[:,0],all_states[:,1],color=blue)
	axis.scatter(all_states[:,2],all_states[:,3],color=green)

if __name__ == "__main__":
	from tp_discretisationmodel import *
	import tp_functions as fn
	def test_eval_value():
		disc = DiscModel()
		model = Model(disc, load_saved = True)
		model.w[disc.target[0]] = -1
		model.w[5 +disc.target[1]] = -1
		model.reward_f = model.buildRewardFunction()
		print model.reward_f
		initial_states = [1,2,3,4,5,6,7,8,9,0,9,8,7,6]
		goal = None;steps = None
		policy,log_policy,z_states=fb.caus_ent_backward(model.transition,model.reward_f,goal,steps,conv=1,z_states = None)
		value = eval_value(model.w,policy,model,initial_states,30)
		print value
	def test_eval_reward():
		disc = DiscModel()
		
		model1 = Model(disc, load_saved = True)
		model1.w[disc.target[0]] = -1
		model1.w[5 +disc.target[1]] = -1
		model1.reward_f = model1.buildRewardFunction()

		model2 = Model(disc, load_saved = True)
		model2.w[disc.target[0]] = -1
		model2.w[5 +disc.target[1]] = -2
		model2.reward_f = model2.buildRewardFunction()
		out = eval_reward_f(model1.reward_f,model2.reward_f)
		print out
	#test_eval_reward()
	plot_state([15,15,1,3],[0,0],[0,0])

	