from sklearn import preprocessing # for data standardization ONLY
import numpy as np

class PCA():
	def __init__(self, components):
		self.components = components

	def transform(self, X):
		# standardize the data
		Xscaled = preprocessing.scale(X) # make each features have mean = 0 and var = 1

		# obtain convariance matrix
		mu = np.mean(Xscaled, axis = 0) # mean sample
		cov = np.cov(Xscaled.T) # numpy cov assumes observations are in columns, and features in the rows, thus we pass X.T instead of X
	
		# obtain eigenvectors and eigenvalues
		eigvals, eigvecs = np.linalg.eig(cov) # eigen decomposition on the covariance matrix

		# sort eigenvectors in decreasing order based on its eigenvalue
		eigens = [(np.abs(val), eigvecs[:, index]) for index, val in enumerate(eigvals)] # list of tuples of each eigenvector with its eigenvalue
		eigens.sort()
		eigens.reverse() 
		
		# select k top vectors to construct the projection matrix W (k = self.components)
		topeigenpairs = eigens[0 : self.components]
		self.W = np.array([eigenpair[1] for eigenpair in topeigenpairs]).T # W is of size |features| x self.components

		# transform X using W to desired dimensionality
		return Xscaled.dot(self.W)

class CrossValidation():
	def __init__(self, model, k = 5, applypca = False, pcadimension = 1):
		self.k = k
		self.model = model
		self.applypca = applypca
		self.pcadimension = pcadimension

	def score(self, X, y):
		# random shuffle
		random = np.random.choice(X.shape[0], X.shape[0], replace = False) 
		data = np.concatenate((X, y), axis = 1)[random]
		
		# partition data into k parts
		#data = np.concatenate((data, np.zeros(shape=(2, X.shape[1] + 1))), axis = 0) # np.split() only works for equal bin division, so we pad the data with dummy data then remove it after partitioning
		self.partitions = np.split(data, self.k)
		self.partitions[self.k - 1] = np.delete(self.partitions[self.k - 1], [self.partitions[self.k - 1].shape[0] - 2, self.partitions[self.k - 1].shape[0] - 1], axis = 0) # remove dummy data
		del data

		# test and train model k times
		self.testacc = []

		for i in xrange(self.k):
			partitions = np.copy(self.partitions)
			Xytest = np.copy(partitions[i])
			partitions = np.delete(partitions, i, axis = 0)
			Xytrain = np.concatenate(partitions, axis = 0)
			
			# seperate features and labels
			Xtrain = Xytrain[:, 0 : Xytrain.shape[1]]
			ytrain = Xytrain[:, Xytrain.shape[1] - 1 : Xytrain.shape[1]]
			Xtest = Xytest[:, 0 : Xytest.shape[1]]
			ytest = Xytest[:, Xytest.shape[1] - 1: Xytest.shape[1]]

			if self.applypca:
				# PCA to reduce dimensionality from 166 to 50
				pca = PCA(self.pcadimension)
				Xtrain = pca.transform(Xtrain)
				Xtest = preprocessing.scale(Xtest).dot(pca.W)

			# fit the model
			self.model.fit(Xtrain, ytrain)

			# test model
			ypredicted = self.model.predict(Xtest)

			# evaluate model and store test error
			self.testacc.append(self.model.accuracy(ytest, ypredicted))

			print str(i) + ' fold complete'

		# determine and return average test error
		return sum(self.testacc) / (1.0 * len(self.testacc))
