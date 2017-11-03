import numpy as np

class MultinomialNaiveBayes():
	def fit(self, X, y):
		'''
		determines the prior and posterior probabilities for a given training data.
		a log transform is used on the prior and posterior probabilities to account 
		for the high feature count causing really small (~ 0) posterior probabilities,
		which make it difficult to predict accurately.
		'''
		# seperate training data by class
		posneg = [[x for x, label in zip(X, y) if label == c] for c in np.unique(y)]
		vocabsize = X.shape[1]

		# obtain prior through finding the proportion of + and - labels in the training set
		self.logprior = np.array([np.log(len(subset) / float(X.shape[0])) for subset in posneg]) # count(yi) / |y|
		
		# obtain posteriors through finding the proportions of Xi|y in the subset of training set with the same corresponding label 
		counts = np.array([np.array(subset).sum(axis = 0) for subset in posneg]) + 1 # add 1 for Laplace smoothing, word counts per class (axis = 0  handle rows)
		labelcounts = counts.sum(axis = 1)[np.newaxis].T # transposing 1D numpy array doesn't work without addition of '[np.newaxis]'
		self.logposterior = np.log((counts * 1.0) / (labelcounts + vocabsize)) # count(wi,yi) / count(yi), add |V| to denominator for Laplace Smoothing  

	def predict(self, X):
		'''
		return predicted class for a given input
		'''
		'''
		# non-log transformation prediction method
		probabilities = []

		for x in X:
			p = np.zeros(shape=(len(X), 1))

			for index, classprob in enumerate(self.posterior):
				numerator = 1
				
				for i, prob in enumerate(classprob):
					numerator *= np.power(prob, x[i])

				p[index] = numerator * self.prior[index]

			probabilities.append(p)
		'''
		probabilities = [(self.logposterior * x).sum(axis = 1) + self.logprior for x in X]

		# return max probability class for each of the test samples
		selections = []
		
		for sample in probabilities:
			index = np.argmax(sample) 
			selections.append([index])
		
		return np.array(selections)

	def accuracy(self, ytest, ypredicted):
		'''
		gives percentage of correct classification
		'''
		correct = 0

		if len(ytest) != len(ypredicted):
			raise ValueError, 'non-matching dimensions'

		for index, y in enumerate(ypredicted):
			if y == ytest[index]:
				correct += 1
		
		return (float(correct) / len(ypredicted)) * 100
