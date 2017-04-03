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
from tp_learn import learn_from_failure

# Experiments go as follows:
# Initialise a discretisation model
# From that Initialise experts using the rewards required for the experiment.
# Initialise test and bad states respectively.
# Make learning runs using different types of apretices i.e that utilise different types of failure (or none)
# Save results in directory.
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
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "slow",initial_bad_states = bad_states)
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")

	def experiment_data_size(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=50,steps=15,runs=6):
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
				results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
				fail[enn,i] = results_failure.e_on_e - results_failure.a_o_e[-1]
				apprentice = Model(disc_a,"uniform", load_saved = True)
				results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
				normal[enn,i] = results_normal.e_on_e - results_normal.a_o_e[-1]
				apprentice = Model(disc_a,"dual_reward", load_saved = True)
				results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "slow",initial_bad_states = bad_states)
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
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "slow",initial_bad_states = bad_states)
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
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()
			apprentice = Model(disc_a,"uniform", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "slow",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")

	def experiment_cvx_contrasting(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=60,steps=15,runs=20):
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
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "L1",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()
			apprentice = Model(disc_a,"uniform", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "false",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			apprentice = Model(disc_a,"dual_reward", load_saved = True)
			results_slow = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = "cvx",initial_bad_states = bad_states)
			if i ==0: 
				apprentice.visualise_reward()			
			results_array.append([results_failure,results_normal,results_slow])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")

	experiment_contrasting(name = "contrasting",steps =15,iterations_per_run= 40,runs = 2)
	experiment_overlapping(name = "overlapping",steps =15,iterations_per_run= 40,runs = 2)
	experiment_complementary(name = "complementary",steps =15,iterations_per_run= 40,runs = 2)

	#experiment_cvx_contrasting(name = "cvx_contrasting",steps =15,iterations_per_run= 40,runs = 2)
	plot_results("cvx_contrasting","results/aamas",2)
	plot_results("complementary","results/aamas",2)
	#plot_results("contrasting","results/aamas",2)
	plot_results("complementary","results/aamas",2)
	#experiment_complementary(name = "complementary",steps =15,iterations_per_run=60 ,runs = 2)
	#xperiment_overlapping(name = "overlapping",steps =15,iterations_per_run= 60,runs = 2)
	#plot_results("complementary","results/aamas",2)
	#plot_results("overlapping","results/aamas",2)









