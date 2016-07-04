import math
import numpy as np
from tp_RFeatures import toy_problem_simple,toy_problem_squared
from tp_kinematics import *
from scipy.stats import norm

class DiscModel(object): # Discretisation for non uniform polar discretisation
	def __init__(self,kinematics = toy_kinematics_gridworld,target = [4,4],boundaries=[4,4],actions = {"linear" :np.array([0,1,2,3,4])},feature = toy_problem_squared):
		self.target = target
		self.boundaries = boundaries		
		robot_x = np.linspace(0,self.boundaries[0],self.boundaries[0]+1) # Nine bins whatever the case
		robot_y = np.linspace(0,self.boundaries[1],self.boundaries[1]+1) # Nine bins whatever the case
		obstacle_v = np.array([0,1,2,3,4])
		obstacle_x = np.linspace(0,self.boundaries[0],self.boundaries[0]+1) # Target orientation bsins
		obstacle_y = np.linspace(0,self.boundaries[1],self.boundaries[1]+1)
		self.feature = feature
		self.actions = actions#these are in the form of a dictionary. but they really should not be TODO
		self.bin_info = np.array([robot_x,robot_y,obstacle_x,obstacle_y,obstacle_v])
		self.kinematics = kinematics #change kinematics
		self.get_dims()
		self.tot_actions = len(self.actions["linear"])
		self.tot_states = np.prod(self.dims)
		self.enumerate_states()
	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = []
		for n,k in enumerate(quantity_vector):
			bins.append(k)
		return map(int,bins)
	def binsToQuantity(self,bins,sample = False):
		quantity = []
		bins = map(int,bins)
		#print bins
		if sample==True:
			for n,k in enumerate(bins):
				if k != (len(self.bin_info[n]) - 1):
					quantity.append(np.random.uniform(self.bin_info[n][k],self.bin_info[n][k+1]))
					#print self.bin_info
				else:
					diff =self.bin_info[n][k] - self.bin_info[n][k-1]  
					quantity.append(np.random.uniform(self.bin_info[n][k], self.bin_info[n][k]+diff))
		else:
			#print sample
			for n,k in enumerate(bins):
				quantity.append(self.bin_info[n][k])

		return quantity
	def binsToState(self,bin_numbers):
		state = 0
		for i in range(0,len(bin_numbers)):
				state+= bin_numbers[i] * np.prod(self.dims[0:i])
		return int(state)
		#Determine where you have ones.
	def stateToBins(self,state):
		#first determine direction.
		bins = np.zeros(len(self.dims))
		for i in range(0,len(self.dims)):
			bins[i] = state%np.prod(self.dims[i])
			state =(math.floor(state/np.prod(self.dims[i])))
		return bins
	def stateToQuantity(self,state):
		bins = self.stateToBins(state)
		return self.binsToQuantity(bins)
	def quantityToState(self,quantity):
		bins = self.quantityToBins(quantity)
		return self.binsToState(bins)
	def actionToIndex(self,action):
		return action
	def indexToAction(self,index):
		return index
	def quantityToFeature(self,state,action=None):
		state = self.stateToQuantity(self.quantityToState(state))
		feature = self.feature(state,self.target,self.boundaries)
		return feature
	def enumerate_states(self):
		self.enum_states = np.zeros([self.tot_states,len(self.stateToQuantity(1))])
		for i in range(self.tot_states):
			self.enum_states[i,:] = self.stateToQuantity(i)

if __name__=="__main__":
	def feature_and_dscretisation_test():
		m = DiscModel(feature = toy_problem_features_simple)
		for i in range(m.tot_states):
			print i
			inp = m.stateToBins(i)
			print inp
			#print "INP",inp
			ou = m.quantityToFeature(inp)
			print np.where(ou==1)
			print len(ou)
			print "_____________________________"
	m = DiscModel()
	#m.indexToAction(0)
	quantity = [0,0,0,0,0]
	state = m.quantityToBins(quantity)
	print state