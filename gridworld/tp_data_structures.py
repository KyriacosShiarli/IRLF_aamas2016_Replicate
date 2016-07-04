import numpy as np

class ExampleStruct(object):
	def __init__(self):
		self.states = []
		self.actions = []
		self.labels = []
		self.state_numbers = []
		self.action_numbers = []
		self.trajectory_number = 0
		self.start = 0
		self.goal = 0
		self.feature_sum = 0
		self.state_action_counts = []
		self.info = ["distance","angle"]

class Results(object):
	def __init__(self,iterations):
		self.test_error= np.zeros(iterations)
		self.train_error = np.zeros(iterations)
		self.train_lik = np.zeros(iterations)
		self.test_lik = np.zeros(iterations)

class EmptyObject(object):
    def __init__(self):
        self.x = None