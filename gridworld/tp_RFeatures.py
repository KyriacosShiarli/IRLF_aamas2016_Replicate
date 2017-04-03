import numpy as np
import math
import pdb
#from discretisation import *
from scipy.stats import norm
def toy_problem_simple(state,target,boundaries):
    # [xtarget,ytarget,xobs,yobs]
    #xdistance from target
    xtar = np.zeros(boundaries[0]+1)
    dist = int(abs(state[0]-target[0]))
    #print "DIST1",dist
    if dist <=boundaries[0]:
        xtar[dist]=1
    else:
        xtar[-1]=1
    #ydistance from target
    ytar = np.zeros(boundaries[1]+1)
    dist = int(abs(state[1]-target[1]))
    #print "DIST2",dist
    if dist <=boundaries[1]:
        ytar[dist]=1
    else:
        ytar[-1]=1
    #xdistance from obstacle
    xobs = np.zeros(boundaries[0]+1)
    dist = int(abs(state[0]-state[2]))
    #print "DIST3",dist
    if dist <=boundaries[0]:
        xobs[dist]=1
    else:
        xobs[-1]=1
    #ydistance from obstacle
    yobs = np.zeros(boundaries[1]+1)
    dist = int(abs(state[1]-state[3]))
    #print "DIST4",dist
    if dist <=boundaries[1]:
        yobs[dist]=1
    else:
        yobs[-1]=1

    return np.concatenate([xtar,ytar,xobs,yobs])
def toy_problem_squared(state,target,boundaries):
    # [xtarget,ytarget,xobs,yobs]
    #xdistance from target
    xtar = np.zeros(boundaries[0]+1)

    arr = []
    for i in range(boundaries[0]+1):
        for j in range(boundaries[1]+1):
            d =np.sqrt(i**2 + j**2)
            arr.append(d)


    unique = np.unique(arr)
    tar_dist = np.zeros(len(unique))
    obs_dist = np.zeros(len(unique))
    #print "DIST1",dist
    dist = np.sqrt((state[0]-target[0])**2 +(state[1]-target[1])**2)
    tar_dist[np.where(unique == dist)[0]] = 1

    dist2 = np.sqrt((state[0]-state[2])**2 +(state[1]-state[3])**2)
    obs_dist[np.where(unique == dist2)[0]] = 1
    #ydistance from target
    return np.concatenate([tar_dist,obs_dist])


if __name__ == "__main__":
    state = [0,0,5,5,4]
    target = [3,3]
    boundaries = [15,15]
    out = toy_problem_features_simple(state,target,boundaries)
    print out
