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

def pin_to_zero(vector_to_pin):
	vector_to_pin = np.array(vector_to_pin)
	where = [i for i in range(len(vector_to_pin)) if vector_to_pin[i] < 0]
	vector_to_pin[where]= 0
	return vector_to_pin
def inference(model,steps,initial_states,discount = 0.9,z_states=None):
	policy,log_policy,z_states=fb.backward_sparse(model.transition.sparse,model.reward_f,length = steps,discount = discount,conv=5,z_states = z_states)
	state_freq,state_action_freq,all_stat = fb.forward_sparse(policy,model.transition.sparse,initial_states,steps,discount=discount) # 30 timesteps
	return policy,z_states,state_action_freq,all_stat
def plot_path(path_probs,disc_model,subplot_dims,save_name):
	fig,axarr = plt.subplots(subplot_dims[0],subplot_dims[1],sharex = False)
	all_states = disc_model.enum_states
	steps = len(path_probs[0,:])
	for i in range(steps):
		alph = abs(path_probs[:,i]-0.001)
		axis = axarr[np.floor(i/subplot_dims[1]),i%subplot_dims[1]]
		green = np.zeros([all_states.shape[0],4]);blue = np.zeros([all_states.shape[0],4])
		blue[:,:3] = np.array([0,0,1]*all_states.shape[0]).reshape(all_states.shape[0],3) ; blue[:,3] = list(alph)
		green[:,:3] = np.array([0,1,0]*all_states.shape[0]).reshape(all_states.shape[0],3) ; green[:,3] = list(alph)
		axis.scatter(disc_model.target[0],disc_model.target[1],color="r",alpha = 0.0001)
		axis.scatter(all_states[:,0],all_states[:,1],color=blue)
		axis.scatter(all_states[:,2],all_states[:,3],color=green)
	fig.savefig(save_name+".png",dpi=80)
def learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = "false",initial_bad_states = None):
	#initialise the lagrange multipliers to 1
	print "INITIALISED LEARNING. LEARNING FROM FAILURE = ",failure
	direc ="results/"
	fn.make_dir(direc)
	C = 5.0
	D=.7
	delta_c = .96
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
	# learning rate
	rate = 0.08
	rate2 = 0.08
	# delay before failure data is includes. Large numbers avoid oscilations
	delay = 0

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
		if failure =="L2":
			#first update the alphas according to their gradient.
			apprentice.w = fn.pin_to_threshold(apprentice.w + rate*difference_exp1,C,-C)		
			if i>delay:
				apprentice.zeta =-difference_exp2
			#print "ZETAAA",apprentice.zeta
			#print "-------------------------------------------"s
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
			delay = 40
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
		apprentice_feature_avg_bad_prev =apprentice_feature_avg_bad 
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


if __name__ == "__main__":
	def experiment_complementary(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=60,steps=15,runs=20):
		direc = "results/aamas"
		#initial_states = [disc.quantityToState([0,0,1,2,2]),disc.quantityToState([0,0,3,4,1]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,2,4,2]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]
		fn.make_dir(direc+"/"+name)
		results_array = []
		disc = DiscModel(target = [4,4],boundaries = [4,4],feature = expert_feature)
		disc_a = DiscModel(target = [4,4],boundaries = [4,4],feature = apprentice_feature)

		expert2 = Model(disc,"obstacle2", load_saved = False)
		expert1 = Model(disc,"target", load_saved = True)
		print "LENl",len(expert1.w)
		test_states = np.random.randint(0,disc.tot_states,10)
		bad_states = np.random.randint(0,disc.tot_states,5)			
		for i in range(runs):
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			initial_states = np.random.randint(0,disc.tot_states,20)
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "sign",initial_bad_states = bad_states)
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")

	def experiment_constrasting(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=50,steps=15,runs=6):
		direc = "results/aamas"
		#initial_states = [disc.quantityToState([0,0,1,2,2]),disc.quantityToState([0,0,3,4,1]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,2,4,2]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]
		fn.make_dir(direc+"/"+name)
		results_array = []
		disc = DiscModel(target = [4,4],boundaries = [4,4],feature = expert_feature)
		disc_a = DiscModel(target = [4,4],boundaries = [4,4],feature = apprentice_feature)
		training_sizes = [2,5,25,50,100]
		fail = np.zeros([len(training_sizes),runs]);normal = np.zeros([len(training_sizes),runs]);slow = np.zeros([len(training_sizes),runs])
		if expert_feature != apprentice_feature:
			expert_2_test = Model(disc,"obstacle2", load_saved = False)
			expert_1_test = Model(disc,"avoid_reach", load_saved = True)
			expert2 = Model(disc_a,"obstacle2", load_saved = False)
			expert2.reward_f = expert_2_test.reward_f
			expert1 = Model(disc_a,"avoid_reach", load_saved = True)
			expert1.reward_f = expert_1_test.reward_f
		else:
			expert2 = Model(disc,"obstacle2", load_saved = False)
			expert1 = Model(disc,"avoid_reach", load_saved = True)
		test_states = np.random.randint(0,disc.tot_states,10)
		bad_states = np.random.randint(0,disc.tot_states,5)	
		for enn,size in enumerate(training_sizes):
			print "SIZE=",size
			print "============================================================================"
			for n,i in enumerate(range(runs)):
				print "RUN",i
				apprentice = Model(disc_a,"dual_reward", load_saved = True)
				#initial_states = np.random.randint(0,disc.tot_states,5)
				initial_states = np.random.randint(0,disc.tot_states,size)
				results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "sign",initial_bad_states = bad_states)
				fail[enn,i] = results_failure.e_on_e - results_failure.a_o_e[-1]
				apprentice = Model(disc_a,"uniform", load_saved = True)
				results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
				normal[enn,i] = results_normal.e_on_e - results_normal.a_o_e[-1]
				apprentice = Model(disc_a,"dual_reward", load_saved = True)
				results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
				slow[enn,i] = results_slow.e_on_e - results_slow.a_o_e[-1]
				results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver((results_array,fail,normal,slow),direc+"/"+name+".pkl")

	def experiment_overlapping(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=60,steps=15,runs=20):
		direc = "results/aamas"
		#initial_states = [disc.quantityToState([0,0,1,2,2]),disc.quantityToState([0,0,3,4,1]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,2,4,2]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]
		fn.make_dir(direc+"/"+name)
		results_array = []
		disc = DiscModel(target = [4,4],boundaries = [4,4],feature = expert_feature)
		disc_a = DiscModel(target = [4,4],boundaries = [4,4],feature = apprentice_feature)

		expert2 = Model(disc,"obstacle2_reach", load_saved = False)
		expert1 = Model(disc,"avoid_reach", load_saved = True)
		test_states = np.random.randint(0,disc.tot_states,10)
		bad_states = np.random.randint(0,disc.tot_states,5)	
		for i in range(runs):
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			initial_states = np.random.randint(0,disc.tot_states,5)
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "sign",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")


	def experiment_contrasting(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=60,steps=15,runs=20):
		direc = "results/aamas"
		#initial_states = [disc.quantityToState([0,0,1,2,2]),disc.quantityToState([0,0,3,4,1]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,2,4,2]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]
		fn.make_dir(direc+"/"+name)
		results_array = []
		disc = DiscModel(target = [4,4],boundaries = [4,4],feature = expert_feature)
		disc_a = DiscModel(target = [4,4],boundaries = [4,4],feature = apprentice_feature)

		expert2 = Model(disc,"obstacle2", load_saved = False)
		expert1 = Model(disc,"avoid_reach", load_saved = True)
		test_states = np.random.randint(0,disc.tot_states,100)
		bad_states = np.random.randint(0,disc.tot_states,5)	
		for i in range(runs):
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			initial_states = np.random.randint(0,disc.tot_states,10)
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "sign",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()
			apprentice = Model(disc_a,"uniform", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")


	def experiment_full(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=60,steps=15,runs=20):
		direc = "results/aamas"
		#initial_states = [disc.quantityToState([0,0,1,2,2]),disc.quantityToState([0,0,3,4,1]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,2,4,2]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]
		fn.make_dir(direc+"/"+name)
		results_array = []
		disc = DiscModel(target = [4,4],boundaries = [4,4],feature = expert_feature)
		disc_a = DiscModel(target = [4,4],boundaries = [4,4],feature = apprentice_feature)

		expert2 = Model(disc,"obstacle2_reach", load_saved = False)
		expert1 = Model(disc,"avoid_reach", load_saved = True)
		test_states = np.random.randint(0,disc.tot_states,100)
		for i in range(runs):
			apprentice = Model(disc_a,"uniform", load_saved = True)
			initial_states = np.arange(0,disc.tot_states)
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false")
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false")		
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false")
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")


		
	#experiment_overlapping(name = "overlapping",steps =15,iterations_per_run= 40,runs = 1)
	#experiment_complementary(name = "overlapping",steps =15,iterations_per_run= 40,runs = 1)
	#experiment_constrasting(name = "contrasting",steps =15,iterations_per_run= 40,runs = 1)
	#plot_results("cvx_contrasting","results/aamas",1)
	#plot_results("complementary","results/aamas",2)
	#plot_results("constrasting","results/aamas",2)
	experiment_contrasting(name = "cvx_sign_contrasting",steps =15,iterations_per_run= 60,runs = 2)
	experiment_complementary(name = "cvx_sign_complementary",steps =15,iterations_per_run=60 ,runs = 2)
	experiment_overlapping(name = "cvx_sign_overlapping",steps =15,iterations_per_run= 60,runs = 2)
	plot_results("cvx_sign_complementary","results/aamas",2)
	plot_results("cvx_sign_contrasting","results/aamas",2)
	plot_results("cvx_sign_overlapping","results/aamas",2)









