import numpy as np
import math
from tp_discretisationmodel import *
import time
import scipy.spatial as spat
from tp_kinematics import toy_kinematics_gridworld
import cPickle as pickle
import tp_functions as fn
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class Transition(object):
	def __init__(self):
		self.forward = []
		self.backward = []
		self.tot_states = 0
		self.tot_actions = 0

class Model(object):
	def __init__(self,discretisation,reward_type,load_saved  = False):
		self.disc = discretisation#discretisation model
		if load_saved == True:
			self.feature_f = fn.pickle_loader("saved/feature_f")
			self.transition = Transition()
			self.transition = fn.pickle_loader("saved/transition_f")
		else:
			self.buildTransitionFunction(1)
			self.buildFeatureFunction()
			fn.pickle_saver(self.feature_f,"saved/feature_f")
			fn.pickle_saver(self.transition,"saved/transition_f")
		self.choose_reward_function(reward_type)
		self.reward_f_initial = self.buildRewardFunction()
		self.reward_f = self.buildRewardFunction()
	def convert_to_sparse(self,inpt):
		tot_states = self.disc.tot_states;tot_actions = self.disc.tot_actions
		#sparse_mat = lil_matrix((tot_states*tot_actions,tot_states))
		sparse_mat = np.zeros((tot_states*tot_actions,tot_states))
		for n,i in enumerate(inpt):
			sparse_mat[n,map(int,i.keys())] = i.values()
		return sparse_mat
	def convert_to_dense(self,inpt):
		num_states = self.disc.tot_states
		num_actions = self.disc.tot_actions
		al = []
		chunks = []
		for i in range(num_states):
			new_state = True
			for j in range(num_actions):
				values = inpt[j+i*num_actions].values()
				keys = map(int,inpt[j+i*num_actions].keys())
				if keys ==[]:
					if new_state:
						chunks.append([0]);chunks[i].append(chunks[i][j])
						new_state = False
					else:
						chunks[i].append(chunks[i][j])
					#print "WARNING"
				if keys !=[]:
					ac = np.ones(len(keys))*j
					col = np.vstack([ac,keys,values])
					if new_state:
						al.append(col)
						new_state = False
						chunks.append([0])
						chunks[i].append(len(keys))
					else:
						al[i] = np.hstack([al[i],col])
						chunks[i].append(len(keys)+chunks[i][j])
		return al,chunks
	def buildTransitionFunction(self,bin_samples):
		self.transition = Transition()
		tot_states=self.transition.tot_states = self.disc.tot_states
		tot_actions=self.transition.tot_actions =self.disc.tot_actions
		self.transition.backward  = [{} for j in xrange(tot_states*tot_actions)];self.transition.forward = [{} for j in xrange(tot_states*tot_actions)]
		for i in xrange(self.disc.tot_states):
			for j in xrange(self.disc.tot_actions):
				bins = self.disc.stateToBins(i)	
				for k in xrange(bin_samples):
					assert bin_samples != 0
					if bin_samples==1:
						samp = False
					else:
						samp = True
					quantity = self.disc.binsToQuantity(bins,sample = samp)
					#print "BINS AND QUANTITY",bins,quantity
					action = self.disc.indexToAction(j)
					next_quantity = self.disc.kinematics(quantity,action,self.disc.boundaries)
					#print "NEXT QUANTITY",next_quantity
					next_state = self.disc.quantityToState(next_quantity)
					if str(next_state) in self.transition.backward[j + i*self.disc.tot_actions].keys():
						self.transition.backward[j + i*self.disc.tot_actions][str(next_state)] += 1.
					else:
						self.transition.backward[j + i*self.disc.tot_actions][str(next_state)] = 1.
				v = self.transition.backward[j + i*tot_actions].values()			
				for key in self.transition.backward[j + i*tot_actions].keys():
					self.transition.backward[j + i*tot_actions][key] /= float(np.sum(v))
					#print "KEY",int(key.split('.')[0])
					self.transition.forward[j + int(key.split('.')[0])*tot_actions][str(i)] = self.transition.backward[j+i*tot_actions][key]
		self.transition.sparse = self.convert_to_sparse(self.transition.backward)
		self.transition.dense_backward,self.transition.chunks_backward = self.convert_to_dense(self.transition.backward)
		self.transition.dense_forward,self.transition.chunks_forward = self.convert_to_dense(self.transition.forward)
	def buildFeatureFunction(self,state_action = True):
		if state_action == True:
			for i in xrange(self.disc.tot_states):
				state_quantity = self.disc.stateToQuantity(i)
				for j in xrange(self.disc.tot_actions):
					action_quantity = self.disc.indexToAction(j)
					features = self.disc.quantityToFeature(state_quantity,action_quantity) #Still hacky. Need to find a solution to his
					if i == 0 and j == 0: feature_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,len(features)])
					feature_f[j,i,:] = features
		elif state_action == False:
			for i in xrange(self.disc.tot_states):
				state_quantity = self.disc.stateToQuantity(i)
				features = self.disc.quantityToFeature(state_quantity) #Still hacky. Need to find a solution to his
				if i == 0: feature_f = np.zeros([self.disc.tot_states,len(features)])
				feature_f[i,:] = features
	 	self.feature_f = feature_f
	def buildRewardFunction(self):
		if hasattr(self,"zeta") == True:
			return np.dot(self.feature_f,self.w +self.zeta)	
		else:
			return np.dot(self.feature_f,self.w)
	def choose_reward_function(self,choice):
		rf = -7*np.ones(self.feature_f.shape[-1])
		if choice == "target":
			if self.disc.feature==toy_problem_squared:
				rf[0] = 0
			elif self.disc.feature==toy_problem_simple:	
				rf[0] = -3;rf[self.disc.boundaries[0]+1] = -3
		elif choice == "obstacle":
			if self.disc.feature==toy_problem_squared:
				rf[16]=0
			elif self.disc.feature==toy_problem_simple:	
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+2] = -1;rf[-self.disc.boundaries[0]-1] = -1
		elif choice == "obstacle2":
			if self.disc.feature==toy_problem_squared:
				rf[15]=0
			elif self.disc.feature==toy_problem_simple:	
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+2] = 1;rf[-self.disc.boundaries[0]-1] = 1
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+3] = -0;rf[-self.disc.boundaries[0]] = -0
		elif choice == "avoid_reach":
			if self.disc.feature==toy_problem_squared:
				rf[0]=0;rf[15]=-20
			elif self.disc.feature==toy_problem_simple:	
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+2] = -10;rf[-self.disc.boundaries[0]-1] = -10
				rf[0] = 0;rf[self.disc.boundaries[0]+1] = 0
		elif choice == "avoid_only":
			if self.disc.feature==toy_problem_squared:
				rf[15]=-20
			elif self.disc.feature==toy_problem_simple:	
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+2] = -20;rf[-self.disc.boundaries[0]-1] = -20
		elif choice == "uniform":
			rf = rf
		elif choice == "dual_reward":
			self.zeta = -3*np.ones(self.feature_f.shape[-1])	

		elif choice == "obstacle2_reach":
			if self.disc.feature==toy_problem_squared:
				rf[15]=-1;rf[0]=0
			elif self.disc.feature==toy_problem_simple:	
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+2] = -1;rf[-self.disc.boundaries[0]-1] = -1 
				rf[self.disc.boundaries[0]+self.disc.boundaries[1]+3] = -7;rf[-self.disc.boundaries[0]] = -7
				rf[0] =1;rf[self.disc.boundaries[0]+1] = 1			
		self.w = rf
		return rf
	def visualise_reward(self):
		fidelity =20

		x = np.linspace(0,4,fidelity)
		y = np.linspace(0,4,fidelity)
		z = np.zeros([fidelity,fidelity])
		rf = np.sum(self.reward_f,axis=0)
		for i in range(fidelity):
			for j in range(fidelity):
				state = [x[i],y[j],2,2,0]
				num = self.disc.quantityToState(state)
				z[i,j] = rf[num]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')		
		X,Y = np.meshgrid(x,y)
		
		surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
		ax.set_xlabel("x-coordinate")
		ax.set_ylabel("y-coordinate")
		ax.set_zlabel("Reward")

	
if __name__ == "__main__":
	d = DiscModel(feature = toy_problem_simple)
	m = Model(d,"avoid_reach",load_saved = False)
	m.visualise_reward()
	m = Model(d,"obstacle2_reach",load_saved = False)
	m.visualise_reward()
	plt.show()
	
	#for i in m.feature_f:
		#print i
	#print d.states_per_angle
	#print m.transition_backward
	#print m.transition_forward
	#print rew
	#print m.transition_f[8,1:30,1:30]
	#st = [1,4,0.0,0.0]
	#action = [0.5,0.44]
	#out = m.kinematics(st,action)
	#print out
