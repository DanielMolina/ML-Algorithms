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
