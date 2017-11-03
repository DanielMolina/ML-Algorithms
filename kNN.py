import numpy as np

class kNN():
	def __init__(self, k = 1):
		self.k = k # number of neighbors to consider

	def distance(self, p, q):
		'''
		euclidean distance between points p and q
		'''
		if len(p) != len(q):
			raise ValueError, 'non-matching dimensions'

		return np.linalg.norm(p - q)

	def fit(self, X, y):
		'''
		learning in kNN just involves storing the representations of the training examples
		'''
		self.X = X
		self.y = y

	def predict(self, X):
		'''
		first, find k nearest points to each test sample in X from random sample of training set instead of every point to lessen runtime.
		finally, assign a test sample the label of the majority of its kNN.
		'''
		ypredicted = np.zeros(len(X))
		data = zip(self.X, self.y) # np.concatenate((self.X, self.y), axis = 1)
		random = np.random.choice(self.X.shape[0], size = 0.1 * self.X.shape[0], replace = False) # consider random 10% sample of training to find kNN, using 100% takes too long
		sample = [data[index] for index in random]

		for index, xtest in enumerate(X):
			distances = sorted((self.distance(xtest, x), y) for x, y in sample)
			knn = distances[0:self.k] # k nearest neignbors
			counts = {}

			# count the number of kNN for each class
			for i in xrange(len(knn)):
				
				number = knn[i][1][0]
				if number not in counts: # new label
					counts[number] = 1
				else: # existing label
					counts[number] += 1

			# determine which label is most prevalent in kNN 
			label, maxcount = None, -1

			for classlabel in counts:
				if counts[classlabel] > maxcount:
					label, maxcount = classlabel, counts[classlabel]

			ypredicted[index] = label
			
		return ypredicted

	def accuracy(self, ytest, ypredicted):
		'''
		gives percentage of correct classification
		'''
		correct = 0

		if len(ytest) != len(ypredicted):
			raise ValueError, 'non-matching dimensions'

		for index, y in enumerate(ypredicted):
			#pdb.set_trace()
			if y == ytest[index]:
				correct += 1
		
		return (float(correct) / len(ypredicted)) * 100
