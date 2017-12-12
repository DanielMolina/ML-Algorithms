import numpy as np 

class kmeans():
	def __init__(self, k = 2, threshold = 0.1, maxiterations = 100):
		self.k = k # num of clusters
		self.threshold = threshold # once the cluster centroids move less than or equal to threshold, stop updating
		self.maxiterations = maxiterations # allows maxiteration number of centroid updatings before stoping 

	def fit(self, Xtrain):
		self.centroids = {} # cluster (key) maps to its centroid

		# initialize centroids with k random points from the training data
		random = np.random.choice(Xtrain.shape[0], self.k, replace = False) # pick k random indicies 

		for k in xrange(self.k):
			self.centroids[k] = Xtrain[random[k]]

		# iterate until convergance or maxiterations reached
		for i in xrange(self.maxiterations):
			self.assignments = {} # cluster (key) maps to list of data points that belong to that cluster

			for cluster in xrange(self.k):
				self.assignments[cluster] = []

			# assign each data point to cluster with closest centroid
			for x in Xtrain:
				distances = [np.linalg.norm(x - self.centroids[centroid]) for centroid in self.centroids] # distance to each of the centroids
				closest = distances.index(min(distances))
				self.assignments[closest].append(x)

			# save previous centroids to check for convergance
			previous = dict(self.centroids)

			# compute new centroid of each cluster by finding the average of all the points in the cluster
			for cluster in self.assignments:
				self.centroids[cluster] = np.average(self.assignments[cluster], axis = 0)

			# update cluster centroids until convergance
			converged = True

			for centroid in self.centroids:
				prev = previous[centroid]
				curr = self.centroids[centroid]

				if np.sum(((curr - prev)/ prev) * 100.0) > self.threshold: # centroids converge iff %-change <= threshold
					converged = False

			if converged:
				break 

		# map each data point to its cluster
		self.members = {}

		for j in xrange(self.k):
			for index, x in enumerate(Xtrain):
				for member in self.assignments[j]:
					if np.array_equal(x, member):
						self.members[index] = j
						break 
		
		# compute objective function value
		ofv = 0
		
		for j in xrange(self.k):
			for index, x in enumerate(Xtrain):
				if self.members[index] == j: # Mi,j = 1
					ofv += np.linalg.norm(x - self.centroids[j])
			
		return ofv

	def predict(self, Xtest):
		assignments = []

		for x in Xtest:
			distances = [np.linalg.norm(x - self.centroids[centroid]) for centroid in self.centroids] # distance to each of the centroids
			closest = distances.index(min(distances))
			assignments.append(closest)

		return assignments
