import tp_forwardBackward as fb
from tp_Model import *
from tp_discretisationmodel import *
import tp_functions as fn
from tp_evaluation import *
import scipy.stats as sps
from tp_data_structures import EmptyObject
#from plots import *
from tp_RFeatures import toy_problem_simple,toy_problem_squared
from tp_plots import plot_results


def inference(model,steps,initial_states,discount = 0.9,z_states=None):
	policy,log_policy,z_states=fb.backward_sparse(model.transition.sparse,model.reward_f,length = steps,discount = discount,conv=5,z_states = z_states)
	state_freq,state_action_freq,all_stat = fb.forward_sparse(policy,model.transition.sparse,initial_states,steps,discount=discount) # 30 timesteps
	return policy,z_states,state_action_freq,all_stat
def learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = "false",initial_bad_states = None):
	#initialise the lagrange multipliers to 1
	print "INITIALISED LEARNING. LEARNING FROM FAILURE = ",failure
	direc ="results/"
	fn.make_dir(direc)
	# learning rate
	rate = 0.08
	rate2 = 0.08
	C = 5.0
	D=.7
	delta_c = .96
	delay = 0
	disc = expert1.disc
	
	a,s,f = expert1.feature_f.shape
	#experts
	exp1_policy,ignore,exp1_state_exp,exp1_all = inference(expert1,steps,initial_states,discount =0.9)
	if initial_bad_states == None:
		exp2_policy,ignore,exp2_state_exp,exp2_all = inference(expert2,steps,initial_states,discount = 0.9)
	else:
		exp2_policy,ignore,exp2_state_exp,exp2_all = inference(expert2,steps,initial_bad_states,discount = 0.9)
	#print "POLICYY", exp1_policy.shape

	exp1_feature_avg = np.dot(exp1_state_exp.reshape(s*a,order = "F"),expert1.feature_f.reshape(s*a,f,order ="F"))
	exp2_feature_avg = np.dot(exp2_state_exp.reshape(s*a,order = "F"),expert2.feature_f.reshape(s*a,f,order = "F"))

	e_on_e = eval_value(expert1.w,exp1_policy,expert1,test_states,steps)
	t_o_t = eval_value(expert2.w,exp2_policy,expert2,test_states,steps)
	expert_on_taboo = eval_value(expert2.w,exp1_policy,expert2,test_states,steps)
	z_stat = None

	#initiate results structure
	results = EmptyObject()
	results.a_o_e = []
	results.a_o_t = []
	results.policy_diff1 = []
	results.policy_diff2 = []
	results.e_on_e = e_on_e
	results.t_o_t = t_o_t
	results.e_o_t = expert_on_taboo


	for i in range(iterations):
		apprentice_policy,z_stat,a_state_exp,a_all = inference(apprentice,steps,initial_states,z_states = None,discount = 0.9)
		apprentice_feature_avg = np.dot(a_state_exp.reshape(s*a,order = "F"),apprentice.feature_f.reshape(s*a,f,order = "F"))
		difference_exp1 = exp1_feature_avg - apprentice_feature_avg
		if initial_bad_states == None:
			difference_exp2 = exp2_feature_avg - apprentice_feature_avg
		else:
			apprentice_policy,z_stat,a_state_exp_bad,a_all = inference(apprentice,steps,initial_bad_states,z_states = None,discount = 0.9)
			apprentice_feature_avg_bad = np.dot(a_state_exp_bad.reshape(s*a,order = "F"),apprentice.feature_f.reshape(s*a,f,order = "F"))
			difference_exp2 = apprentice_feature_avg_bad - exp2_feature_avg 
		if i ==0:
			difference_random = np.copy(difference_exp2)
			apprentice_feature_avg_bad_prev = apprentice_feature_avg*0
		#updates
		elif failure == "L1":
			apprentice.w = apprentice.w + rate*difference_exp1
			#apprentice.zeta = apprentice.zeta + rate2*(difference_exp2+ D*apprentice.zeta)
			apprentice.zeta = 0.9*difference_exp2
		elif failure == "false":
			apprentice.w = apprentice.w + rate*difference_exp1
		elif failure == "slow":
			apprentice.w = apprentice.w + rate*difference_exp1	
			C = C*delta_c
			if 1./C>D:
				C = 1/D
			if i >delay:
				apprentice.zeta =-difference_exp2/(C)
			#print "ZETAAA",apprentice.zeta
			#print "-------------------------------------------"
		elif failure == "cvx":	
			delay = 0
			apprentice.w = apprentice.w + rate*difference_exp1
			#sings = difference_random*difference_exp2
			#print sings
			#idx = np.where(sings < 0)
			#difference_exp2[idx]=0
			rho = 0.01
			#if rho>0.8:
			#	rho=0.8
			#apprentice.zeta = apprentice.zeta + rate2*(difference_exp2+ D*apprentice.zeta)
			if i>delay:
				apprentice.zeta =0.9*(apprentice_feature_avg_bad_prev - rho*apprentice_feature_avg_bad + (rho-1)*exp2_feature_avg) 
			apprentice_feature_avg_bad_prev =apprentice_feature_avg_bad 
			#apprentice.zeta = difference_random - 0.2*difference_exp2
		elif failure == "sign":	
			apprentice.w = apprentice.w + rate*difference_exp1
			rho = 0.01
			apprentice.zeta =np.sign(difference_random)
		elif failure == "only":
			apprentice.zeta =apprentice.zeta -rate2*(difference_exp2 + D*apprentice.zeta)
			apprentice.zeta = -2*difference_exp2
			#print "ZETAAA",apprentice.zeta
			#print "-------------------------------------------"
		
		apprentice.reward_f = apprentice.buildRewardFunction()
		#evaluation
		a_on_e = eval_value(expert1.w,apprentice_policy,apprentice,test_states,steps) 
		a_o_t = eval_value(expert2.w,apprentice_policy,apprentice,test_states,steps) 
		#if i ==iterations-1:
		if i <iterations:			
			print "failure",failure
			print "Iteration",i
			print "Aprentice on Expert" ,a_on_e
			print "Expert on expert",e_on_e
			print "Apprentice on Taboo",a_o_t
			print "Taboo on Taboo",t_o_t
			print "Expert on Taboo",expert_on_taboo
			print "______________________________________"
		results.a_o_e.append(a_on_e)
		results.a_o_t.append(a_o_t)
		results.policy_diff1.append(np.sum(np.sum(np.absolute(apprentice_policy-exp1_policy)))/(2*disc.tot_states)*100)
		results.policy_diff2.append(np.sum(np.sum(np.absolute(apprentice_policy-exp2_policy)))/(2*disc.tot_states)*100)
		if i  == iterations-1:
			print "Policy Difference",results.policy_diff1[-1]
			print "Policy Difference",results.policy_diff2[-1]
	return results









