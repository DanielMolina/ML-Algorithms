import numpy as np

class LogisticRegression():
	def __init__(self, learningrate = 1.0, threshold = 0.1, maxiterations = 50):
		self.learningrate = learningrate
		self.maxiterations = maxiterations
		self.threshold = threshold

	def logisticfunc(self, wx):
		return 1.0 / (1.0 + np.exp(-wx))

	def fit(self, X, y):
		# add N x 1 column of 1s to front of X to account for w0
		X = np.insert(X, 0, 1, axis = 1)

		# initialize the w vector
		self.w = np.ones(X.shape[1])[np.newaxis].T
		
		# weight update, threshold not utilized 
		for iteration in range(self.maxiterations):
			predicted = X.dot(self.w) # w0 + w1x1 + ... per sample
			error = y - self.logisticfunc(predicted) # y - P(y|x;w)
			self.w += self.learningrate * (X.T).dot(error) # w = w + n[X^T(y - P(y|X;w))]
		'''
		# weight update, usual algorithmic implementation
		for iteration in range(self.maxiterations):
			oldw = self.w

			for index, wi in enumerate(self.w):
				error = 0

				for i, x in enumerate(X):
					predicted = x[np.newaxis].dot(self.w)
					error += x[index] * (y[i] - self.logisticfunc(predicted))

				self.w[index] = wi + (self.learningrate * error)

			# break if change is less than threshold 
			changes = self.w - oldw
			divergance = [change if change > self.threshold for change in changes]
			
			if len(divergance) == 0:
				break 
		'''

	def predict(self, X):
		'''
		return predicted class for a given input
		'''
		# add N x 1 column of 1s to front of X to account for w0
		X = np.insert(X, 0, 1, axis = 1)

		predicted = X.dot(self.w)
		logisticpred = self.logisticfunc(predicted) 

		# if mu(wx) < 0.5, label is 0, else, label is 1
		return np.array([0.0 if label[0] < 0.5 else 1.0 for label in logisticpred])

	def accuracy(self, ytest, ypredicted):
		'''
		gives percentage of correct classification
		'''
		correct = 0

		if len(ytest) != len(ypredicted):
			raise ValueError, 'non-matching dimensions'
		
		for index, y in enumerate(ypredicted):
			if y == ytest[index][0]:
				correct += 1
		
		return (float(correct) / len(ypredicted)) * 100
