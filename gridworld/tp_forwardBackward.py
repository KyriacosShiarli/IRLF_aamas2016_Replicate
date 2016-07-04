import sys
import numpy as np
import math
import random as rd
import time
from tp_functions import sum_chunks,discounted_sum,timer

def forward(policy,transition,start,time_steps, discount = None):
    #print "START of FORWARD ------------------->"
    num_actions = transition.tot_actions;num_states = transition.tot_states
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states_actions = np.zeros([num_actions,num_states,time_steps])
    for i in start: dt_states[i,0]+=1 
    dt_states[:,0] /=len(start)
    for i in range(time_steps):
      for j in range(num_states):
        tr = transition.dense_forward[j]
        if i != time_steps-1:
          dt_states[j,i+1] = np.sum(dt_states[map(int,tr[1,:]),i] *policy[map(int,tr[0,:]),map(int,tr[1,:])] * tr[2,:]) 
        dt_states_actions[:,j,i] = dt_states[j,i]*policy[:,j]
    if discount ==None:
      state_action_freq = np.sum(dt_states_actions,axis=2)
      state_freq = np.sum(dt_states,axis = 1)
    else: 
      state_action_freq = discounted_sum(dt_states_actions,discount,ax=2)
      state_freq = discounted_sum(dt_states,discount,ax = 1)
    
    #print "END of Forward"
    return state_freq,state_action_freq,dt_states

def forward_sparse(policy,transition_forward,start,time_steps, discount = None):
    #print "START of FORWARD ------------------->"
    num_states = transition_forward.shape[1];num_actions = transition_forward.shape[0]/num_states
    #print num_states,num_actions
    #alpha = 1w*0/(num_states*100)
    alpha = 0
    dt_states = np.zeros((num_states,time_steps))
    dt_states_actions = np.zeros((num_actions*num_states,time_steps))
    for i in start: dt_states[i,0]+=1 
    dt_states[:,0] /=len(start)
    for i in range(time_steps):
      #print "CHAPE YOU WANT",(dt_states[:,i]*policy).shape
      dt_states_actions[:,i] = (dt_states[:,i]*policy).reshape(num_actions*num_states,order="F").T
      if i != time_steps-1:
        dt_states[:,i+1] =np.dot((1-alpha)*transition_forward.T,dt_states_actions[:,i]) + alpha*np.sum(dt_states_actions[:,i]) 
    if discount ==None:
      state_action_freq = np.sum(dt_states_actions,axis=1)
      state_freq = np.sum(dt_states,axis = 1)
    else: 
      state_action_freq = discounted_sum(dt_states_actions,discount,ax=1)
      state_freq = discounted_sum(dt_states,discount,ax = 1)
    state_action_freq = state_action_freq.reshape(num_actions,num_states,order = "F")
    return state_freq,state_action_freq,dt_states


def backward_sparse(sparse_transition,reward_f,conv=5,discount = 1.,length = 15,z_states = None):
    num_states = sparse_transition.shape[1]; num_actions = sparse_transition.shape[0]/num_states
    #if reward_f.shape[0] ==num_actions:
    #  state_action = True
    #else: state_action =False
    z_actions = np.zeros(num_actions*num_states)
    if z_states==None:
      z_states = np.zeros(num_states)
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    #print "Caus Ent Backward"
    count = 0
    delta = 0
    reward_temp = reward_f.reshape(num_actions*num_states,order="F")
    #while True:
    alpha = (1/(num_states*100))*0
    for i in range(length):
      prev = np.zeros(z_states.shape)
      prev += z_states
      #print gamma*sparse_transition.dot(z_states)
      z_actions = discount*((1-alpha)*sparse_transition.dot(z_states) + alpha*np.sum(z_states) ) +reward_temp
      m = np.amax(z_actions.reshape(num_actions,num_states,order="F"),axis = 0)
      z_states = m + np.log(np.sum(np.exp(z_actions.reshape(num_actions,num_states,order="F")-m),axis = 0))
      count+=1
      #Action Probability Computation - - - - - - - - - - - - - - - -
      delta = np.amax(np.absolute(prev-z_states))
      #print delta
      #if count>2 and delta<conv:
        #print "Count and delta", count,delta
      policy= np.exp(z_actions.reshape(num_actions,num_states,order="F")-z_states)
        #print "Policyyy",policy
        #break
    return policy,np.log(policy),z_states




def caus_ent_backward(transition,reward_f,conv=5,discount = 0.9,z_states = None):
    num_actions = transition.tot_actions;num_states = transition.tot_states
    if reward_f.shape[0] ==num_actions:
      state_action = True
    else: state_action =False
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = np.zeros(num_states)
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    #print "Caus Ent Backward"
    count = 0
    delta = 0
    while True:
      prev = np.zeros(z_states.shape)
      prev += z_states
      for i in range(num_states):
        tr = transition.dense_backward[i]
        ch = transition.chunks_backward[i]
        out = discount*np.array(sum_chunks(tr[2,:]*z_states[map(int,tr[1,:])],ch))
        z_actions[:,i] = out +reward_f[:,i]
      m = np.amax(z_actions,axis = 0)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      count+=1
      #Action Probability Computation - - - - - - - - - - - - - - - -
      delta = np.amax(np.absolute(prev-z_states))
      if count == 50:
        #print "Count and delta", count,delta
        z_actions = z_actions
        m = np.amax(z_actions,axis = 0)
        z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
        policy= np.exp(z_actions-z_states)
        break
    return policy,np.log(policy),z_states

def timed_backward(transition,reward_f,conv=5,discount = 0.9,z_states = None):
    sweep_timer = timer()
    convergence_timer = timer()
    num_actions = transition.tot_actions;num_states = transition.tot_states
    if reward_f.shape[0] ==num_actions:
      state_action = True
    else: state_action =False
    gamma = discount
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = np.zeros(num_states)
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Caus Ent Backward"
    count = 0
    delta = 0
    convergence_timer.start()
    while True:
      prev = np.zeros(z_states.shape)
      prev += z_states
      sweep_timer.start()
      for i in range(num_states):
        tr = transition.dense_backward[i]
        ch = transition.chunks_backward[i]
        out = gamma*np.array(sum_chunks(tr[2,:]*z_states[map(int,tr[1,:])],ch))
        z_actions[:,i] = out +reward_f[:,i]
      sweep_timer.stop()
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      count+=1
      #Action Probability Computation - - - - - - - - - - - - - - - -
      delta = np.amax(np.absolute(prev-z_states))
      print delta
      if count>2 and delta<conv:
        print "Count and delta", count,delta
        z_actions = z_actions
        m = np.amax(z_actions)
        z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
        policy= np.exp(z_actions-z_states)
        break
    convergence_timer.stop()
    times = [sum(convergence_timer.time_taken)/len(convergence_timer.time_taken),sum(sweep_timer.time_taken)/len(sweep_timer.time_taken)]
    return policy,np.log(policy),z_states,times

def caus_ent_backward_nodisount(transition,reward_f,steps):
    num_actions = transition.tot_actions;num_states = transition.tot_states
    if reward_f.shape[0] ==num_actions:
      state_action = True
    else: state_action =False
    gamma = discount
    z_actions = np.zeros([num_actions,num_states])
    z_states = np.zeros(num_states)
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Caus Ent Backward"
    count = 0
    delta = 0
    for j in range(steps):
      prev = np.zeros(z_states.shape)
      prev += z_states
      for i in range(num_states):
        tr = transition.dense_backward[i]
        ch = transition.chunks_backward[i]
        out = gamma*np.array(sum_chunks(tr[2,:]*z_states[map(int,tr[1,:])],ch))
        z_actions[:,i] = out +reward_f[:,i]
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      count+=1
      #Action Probability Computation - - - - - - - - - - - - - - - -
      delta = np.sum(np.sum(np.absolute(prev-z_states)))
      #delta +=1 
      #print "DElta cause",delta,delta2
      if j==steps-1:
        z_actions = z_actions
        m = np.amax(z_actions)
        z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
        policy= np.exp(z_actions-z_states)
    return policy,np.log(policy),z_states


'''
if __name__=="__main__":
  from discretisationmodel import *
  from Model import Model
  from discretisationmodel import 
  disc_model = DiscModel()
  w = None
  model = Model(disc_model,w,learn = False)
  p,lp,zs = caus_ent_backward(model.transition,model.reward_f,1,1,6,z_states=None)
  p2,lp2 = caus_ent_backward_test(model.transition,model.reward_f,1,1,6,z_states=None)
  print "FIRST DIFFERENCE", p-p2
  state_frequencies,sa= forward(p,model.transition,[1,2,3,4,5,6],5)
  state_frequencies_t,sa_t= forward_test(p,model.transition,[1,2,3,4,5,6],5)
'''
if __name__ == "__main__":
  from tp_discretisationmodel import *
  import matplotlib.pyplot as plt
  from tp_Model import *

  def test_toy_problem_inference():
    d = DiscModel()
    m = Model(d,load_saved = True)
    goal = 0
    steps = 30
    p,np,z=caus_ent_backward(m.transition,m.reward_f,goal,steps,conv=100,z_states = None)
    state_freq,state_action_freq = forward(p,m.transition,[1,2,3,4],30)
    #print "Difference", sa_t - sa
  test_toy_problem_inference()

