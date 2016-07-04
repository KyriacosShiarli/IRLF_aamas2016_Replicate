import matplotlib.pyplot as plt
import tp_functions as fn
import pdb
from tp_data_structures import EmptyObject
import numpy as np

class Empty(object):
	def __init__(self):
		self.a_o_e = []
		self.a_o_t = []
		self.e_on_e = []
		self.t_o_t = []
		self.e_o_t = []
		self.policy_diff1 = []
		self.policy_diff2 = []
def results_plot_individual_means(results1,results2,results3,name,num_of_runs = 10):
	f,axarr = plt.subplots(1,1,sharex=False)
	iterations = len(results1.mean_a_o_e)
	x = range(iterations)
	axarr.errorbar(x,results1.mean_a_o_e,yerr = results1.std_a_o_e/np.sqrt(num_of_runs),color = "b",marker = '.',label= "Convex")
	axarr.errorbar(x,results2.mean_a_o_e,yerr = results2.std_a_o_e/np.sqrt(num_of_runs),color = "r",marker = '.',label= "IRL")
	axarr.errorbar(x,results3.mean_a_o_e,yerr = results3.std_a_o_e/np.sqrt(num_of_runs),color = "g",marker = '.',label= "IRLF (previous)")
	axarr.plot(x,[results1.e_on_e]*iterations,color = "k",label = "Successful Test Data",linewidth=2)
	axarr.set_ylabel("Value")
	axarr.set_xlim((0,iterations+5))
	axarr.legend(bbox_to_anchor=(1.0, 0.5,0.,0.0),loc=1, prop={"size":16})
	axarr.set_xlabel("iterations")
	f.set_size_inches(6.5,4.5)
	f.savefig(name+'/expert_apprentice.png',dpi=80)
	f.savefig(name+'/expert_apprentice.pdf',dpi=80)
	f,axarr = plt.subplots(1,1,sharex=False)
	axarr.set_ylabel("Value")
	axarr.set_xlim((0,iterations+5))
	axarr.errorbar(x,results1.mean_a_o_t,yerr = results1.std_a_o_t/np.sqrt(num_of_runs),color = "b",marker = '.',label= "Convex")
	axarr.errorbar(x,results2.mean_a_o_t,yerr = results2.std_a_o_t/np.sqrt(num_of_runs),color = "r",marker = '.',label= "IRL")
	axarr.errorbar(x,results3.mean_a_o_t,yerr = results3.std_a_o_t/np.sqrt(num_of_runs),color = "g",marker = '.',label= "IRLF (previous)")
	#axarr[1].plot(x,[results.t_o_t]*iterations,color = "m",label = "non expert a_on_e")
	axarr.plot(x,[results1.e_o_t]*iterations,color = "k",label = "Successful Test Data",linewidth=2)
	axarr.legend(bbox_to_anchor=(1.0, 1.0,0.,-0.5),loc=1, prop={"size":13})
	axarr.set_xlabel("iterations")
	f.set_size_inches(6.5,4.5)
	f.savefig(name+'/taboo_apprentice.png',dpi=80)
	f.savefig(name+'/taboo_apprentice.pdf',dpi=80)

	f,axarr = plt.subplots(1,1,sharex=False)
	axarr.errorbar(x,results1.mean_policy_diff1,yerr = results1.std_policy_diff1,color ="b",label = "IRLF" )
	axarr.errorbar(x,results2.mean_policy_diff1,yerr = results2.std_policy_diff1,color ="r",label = "MaxEnt" )
	axarr.set_xlabel("iterations")
	axarr.set_ylabel("Policy Difference")
	
	f.savefig(name+'/policy_diff_expert.png',dpi=80)


def average_and_std_type(all_results,index):
	avg = EmptyObject()

	for n,run in enumerate(all_results):
		if n ==0:
			avg.a_o_e = np.array(run[index].a_o_e); avg.a_o_t = np.array(run[index].a_o_t)
			avg.policy_diff1 = run[index].policy_diff1; avg.policy_diff2 = run[index].policy_diff2
			avg.e_on_e = [run[index].e_on_e]; avg.e_o_t = [run[index].e_o_t]
		else:
			avg.a_o_e = np.vstack((avg.a_o_e,np.array(run[index].a_o_e))); avg.a_o_t =np.vstack((avg.a_o_t, np.array(run[index].a_o_t)))
			avg.policy_diff1 =np.vstack((avg.policy_diff1, np.array(run[index].policy_diff1))); avg.policy_diff2 =np.vstack((avg.policy_diff2,np.array(run[index].policy_diff2)))
			avg.e_on_e.append(run[index].e_on_e); avg.e_o_t.append(run[index].e_o_t)
	avg.mean_a_o_e  = np.mean(avg.a_o_e,axis = 0); avg.std_a_o_e = np.std(avg.a_o_e,axis = 0)
	avg.mean_a_o_t  = np.mean(avg.a_o_t,axis = 0); avg.std_a_o_t = np.std(avg.a_o_t,axis = 0)
	avg.mean_policy_diff1 = np.mean(avg.policy_diff1,axis=0); avg.std_policy_diff1  = np.std(avg.policy_diff1,axis = 0)
	avg.mean_policy_diff2 = np.mean(avg.policy_diff2,axis=0); avg.std_policy_diff2  = np.std(avg.policy_diff2,axis = 0)
	avg.e_on_e = np.mean(avg.e_on_e); avg.e_o_t = np.mean(avg.e_o_t)
	return avg

def plot_results(experiment_name, directory,number_of_runs):
	all_results = fn.pickle_loader(directory+"/"+experiment_name+".pkl")
	print len(all_results[0])
	results_f = average_and_std_type(all_results,0)
	results_n = average_and_std_type(all_results,1)
	results_s = average_and_std_type(all_results,2)
	results_plot_individual_means(results_f,results_n,results_s,directory+"/"+experiment_name,num_of_runs=number_of_runs)























