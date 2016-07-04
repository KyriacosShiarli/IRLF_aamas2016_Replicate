
import os
from itertools import compress
import math
import numpy as np
from scipy.stats import norm
import scipy.ndimage.filters as filt
import cPickle as pickle
import time

def getFolds(examples,percent,idx):
	k = int(np.floor(len(examples)*percent))
	ran = range(0,len(examples),k)
	mask = [1]*len(examples)
	mask[ran[idx]:ran[idx]+k] = [0]*k
	test = examples[ran[idx]:ran[idx]+k]
	train = list(compress(examples,mask)) 
	return train,test
def momentum(current,previous,decay):
	new = current + decay*previous
	return new
def angle_half_to_full(inp):
	#transforms vector from half angle mode to full angle mode
	assert inp>=-math.pi,"not an angle %s" %inp  ; assert inp<=math.pi,"not an angle %s" %inp 
	if inp <0:
		inp += 2*math.pi
	return inp

def angle_full_to_half(inp):
	#transforms vector from half angle mode to full angle mode
	assert inp<=2*math.pi,"not an angle %s" %inp ; assert inp>=0,"not an angle %s" %inp
	if inp>math.pi:
		inp -= 2*math.pi
	return inp
def vec_full_to_half(in_vec):
	out = np.zeros(len(in_vec))
	for n,i in enumerate(in_vec):
		out[n] = angle_full_to_half(i)
	return out
def vec_half_to_full(in_vec):
	out = np.zeros(len(in_vec))
	for n,i in enumerate(in_vec):
		out[n] = angle_half_to_full(i)
	return out
def polar_to_cartesian(theta,distance):
	x = distance * np.cos(theta)
	y = distance * np.sin(theta)
	return x,y
def cartesian_to_polar(x,y):
	distance = np.sqrt(x*x + y*y)
	theta = np.array([arctan_correct(x[i],y[i]) for i in xrange(len(x))])
	return np.array(zip(theta,distance))
def c_o_m(persons_in_polar):
	#print persons_in_polar[0][:,1]
	cartesian =np.array([np.array(polar_to_cartesian(p[:,0],p[:,1])) for  p in persons_in_polar])
	car_com = np.sum(cartesian,axis = 0)/len(persons_in_polar)
	return cartesian_to_polar(car_com[0],car_com[1])
def arctan_correct(x,y):
	if x<0 and y<0:
		return np.arctan(y/x)- math.pi
	if x<0 and y>0:
		return np.arctan(y/x)+math.pi
	else:
		return np.arctan(y/x)

def subsample(sample_vector,factor,scal = 0.0001):
	length = factor
	rv = norm(loc = 0., scale = scal)
	le = float(len(sample_vector))
	if (le/factor  - int(le/factor))<0.5 and (le/factor  - int(le/factor))!=0.0 :
		center_idx = np.arange(factor,le-factor,factor)
		new_vec = np.zeros(int(le/factor))
	elif (le/factor  - int(le/factor))==0.0:
		center_idx = np.arange(factor,le,factor)
		new_vec = np.zeros(int(le/factor))
	else:
		center_idx = np.arange(factor,le,factor)
		new_vec = np.zeros(int(le/factor)+1)
	new_vec[0]= sample_vector[0]
	for n, i in enumerate(center_idx):
		div =0
		if n == 0 or n == len(center_idx)-1:
			length = factor
		else:
			length = factor*2
		for j in range(length):
			new_vec[n+1]+=sample_vector[i-int(length/2)+j] *  np.exp(-((-int(length/2)+j)**2./scal))
			div += np.exp(-((-int(length/2)+j)**2./scal))
		new_vec[n+1]/=div
	return new_vec

def angle_smoother(angle_vector,filter_scale):
	#In radians angles cannot be smoothed. They are converted to cartesian smoothed and returned.
	vector = np.array([list(polar_to_cartesian(thet,1)) for thet in angle_vector])
	vector[:,0] = filt.gaussian_filter(vector[:,0],filter_scale)
	vector[:,1] = filt.gaussian_filter(vector[:,1],filter_scale)
	out  = cartesian_to_polar(vector[:,0],vector[:,1])
	#out = np.array([list(cartesian_to_polar(i[0],i[1])) for i in vector])
	return out[:,0]

def angle_subsampler(angle_vector,subsample_factor,scal = 1):
	vector = np.array([list(polar_to_cartesian(thet,1)) for thet in angle_vector])
	out =cartesian_to_polar(subsample(vector[:,0],subsample_factor,scal), subsample(vector[:,1],subsample_factor,scal))
	return out[:,0]
def ang_vel(yaw1,yaw2,duration):
	if (yaw2 - yaw1) > 2*math.pi-0.2:
		diff = (yaw2 - yaw1) - 2*math.pi
	elif (yaw2 - yaw1) < -2*math.pi+0.2:
		diff = (yaw2 - yaw1) + 2*math.pi
	else:
		diff = (yaw2 - yaw1)
	if np.absolute(diff) >15:
		print diff
		print yaw1,yaw2 
	return diff/duration
def make_dir(directory):
  	if not os.path.exists(directory):
  		os.makedirs(directory)

def distance_function(traj1,traj2):
	x1,y1 = polar_to_cartesian(traj1[:,0],np.ones(traj1.shape[0]))
	x2,y2 = polar_to_cartesian(traj2[:,0],np.ones(traj1.shape[0]))
	x1_target,y1_target = polar_to_cartesian(traj1[:,2],np.ones(traj1.shape[0]))
	x2_target,y2_target = polar_to_cartesian(traj2[:,2],np.ones(traj2.shape[0]))

	dist_vector = [np.absolute(x1-x2),np.absolute(y1-y2),np.absolute(traj1[:,1]-traj2[:,1]),np.abs(x1_target-x2_target)
					,np.abs(y1_target-y2_target),np.absolute(traj1[:,3]-traj2[:,3]),np.absolute(traj1[:,4]-traj2[:,4]),np.absolute(traj1[:,5]-traj2[:,5])]
	range_normalisation = np.array([[1.,1.,3,1.,1.,4,0.6,1.]])
	distance = np.sum(np.sum(dist_vector))/traj1.shape[0]
	return distance

def trajectory_to_cartesian(trajectory):
	length = trajectory.shape[0]
	x1,y1 = polar_to_cartesian(trajectory[:,0],np.ones(length))
	xt,yt = polar_to_cartesian(trajectory[:,2],np.ones(length))
	car_trajectory = np.vstack([x1,y1,trajectory[:,1],xt,yt,trajectory[:,3]])
	return np.transpose(car_trajectory)

def state_to_cartesian(polar_state):
	one = np.array([1])
	out1 = np.array(list(polar_to_cartesian(np.array([polar_state[0]]),one)))
	outt  = np.array(list(polar_to_cartesian(np.array([polar_state[2]]),one)))
	return np.array([out1[0,0],out1[1,0],polar_state[1],outt[0,0],outt[1,0],polar_state[3]])

def state_to_polar(cartesian_state):
	theta_group = cartesian_to_polar(np.array([cartesian_state[0]]),np.array([cartesian_state[1]]))
	theta_group = theta_group[0,0]
	theta_target = cartesian_to_polar(np.array([cartesian_state[3]]),np.array([cartesian_state[4]]))
	theta_target = theta_target[0,0]
	return np.array([theta_group,cartesian_state[2],theta_target,cartesian_state[5]])

def pickle_saver(to_be_saved,full_directory):
	with open(full_directory,'wb') as output:
		pickle.dump(to_be_saved,output,-1)

def pickle_loader(full_directory):
	with open(full_directory,'rb') as input:
		return pickle.load(input)

def discounted_sum(array_to_sum,factor,ax=0):
	assert factor<=1 #if factor addition this is simply the sum
	#Input must be a numpy array
	if factor ==1:
		out = np.sum(array_to_sum,axis=ax)
		return out
	elif len(array_to_sum.shape)==1:# single dimentional array
		out = 0
	elif len(array_to_sum.shape)>1:
		mask = np.ones(len(array_to_sum.shape),dtype=bool)
		mask[ax] = False
		#print "IN DISCOUNTER----------------------"
		#print "MASK", mask
		#print array_to_sum.shape
		outdim = np.array(array_to_sum.shape)[mask]
		dims = np.array(range(len(array_to_sum.shape)))[mask]
		#print ax,dims
		out = np.zeros(outdim)
		#print "TRANSPOSITIONS", np.hstack([[ax],dims])
		array_to_sum = array_to_sum.transpose(np.hstack([[ax],dims]))
	for i,j in enumerate(array_to_sum):
		out +=j*factor**i
	return out

def pin_to_threshold(vector_to_pin,maximum,minimum):
	assert maximum > minimum
	vector_to_pin = np.array(vector_to_pin)
	where_max = [i for i in range(len(vector_to_pin)) if vector_to_pin[i] > maximum]
	where_min = [i for i in range(len(vector_to_pin)) if vector_to_pin[i] < minimum]
	vector_to_pin[where_max]= maximum
	vector_to_pin[where_min]= minimum
	return vector_to_pin

class timer(object):
	def __init__(self):
		self.start_time = []
		self.end_time = []
		self.time_taken = []
	def start(self):
		self.start_time.append(time.clock())
	def stop(self):
		self.end_time.append(time.clock())
		self.time_taken.append(self.end_time[-1]-self.start_time[-1])
def sum_chunks(one_d_arr,chunks):
	out = [np.sum(one_d_arr[chunks[i]:chunks[i+1]]) for i in range(len(chunks)-1)]
	return out
if __name__ == "__main__":
	def test_folds():
		examples = load_all(30)
		print len(examples)
		for i in range(4):
			train,test = getFolds(examples,0.25,i)
			print len(train),len(test)

	def test_discounter():
		arr = np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]])
		out = discounted_sum(arr,0.8,ax=1)
		print out
	def test_pickle_saver_loader():
		x = 5 
		directory = "TESTS/funtions/picklesaveload/"
		make_dir(directory)
		x = 5
		pickle_saver(x,directory+"test.pkl")
		ex = pickle_loader(directory+"test.pkl")
		if x - ex ==0:
			print "PICKLE TEST PASSED"
		else: 
			print "PIACLE TEST FAILED" 
	#test_discounter()
	def test_threshold_pin():
		vector = [1,2,5,-15,35]
		test = [1,2,5,-5,5]
		out = pin_to_threshold(vector,5,-5)
		if sum(test-out)==0:
			print "passed",out,test
	test_threshold_pin()

	# test_pickle_saver_loader()
	# test_folds()
	# traj1 = np.ones([350,6])
	# traj2 = np.ones([350,6])
	# traj1[1,1] = 50
	# dist = distance_function(traj1,traj2)
	# print dist
	#test =  np.arange(-math.pi,math.pi,0.1)
	#ar = [1,4,5,6,7,8,9,9,9]
	#ch = [0,2,9]
	#print np.sum(ar[2:9])
	#out = sum_chunks(ar,ch)
	#print out
	# x = np.linspace(-math.pi,math.pi,14)
	# angle_smoother(x,50)



	#out = map (angle_half_to_full,test)
	#out2 = map (angle_full_to_half,out)
	#print test - out2
	#for i in range(100,500):
	#	print i
	#	vector = np.arange(0,i,1)
	#	out = subsample(vector,10)
	#	print out

