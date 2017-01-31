#!/Users/me/anaconda2/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Plot data
def plotDataSet(X):
	plt.clf()
	plt.title("PCA data")
	plt.scatter(X[:,0], X[:,1], c="b", marker='o')
	plt.xlim(X.min(axis=0)[0], X.max(axis=0)[0])
	plt.ylim(X.min(axis=0)[1], X.max(axis=0)[1])
	plt.xlabel('Fahrenheit')
	plt.ylabel('Celsius')
	plt.show()

# Create a set for demonstrating the PCA method
# The function creates points along the y=x and then offsets them 
# on a direction perpendicular to the main diagonal
# The offset strength is controlled by the variable variance
# Variable _size defines the number of points to be created
def createPCASet(_size, variance=1.0):
	x = np.linspace(0,50, num=_size)
	data = np.c_[x,x]
	v = (np.random.ranf(_size)-0.5)*variance
	offsets = np.c_[-v,44.8*v+5]
	data += offsets

	return data

def createFahreneitCelciusSet(_size, variance=1.0):
	v = (np.random.ranf(_size)-0.5)*variance
	x0 = np.linspace(0,100, num=_size) + v
	v = (np.random.ranf(_size)-0.5)*variance
	x1 = (x0-32.0)*5.0/9.0 + v
	data = np.c_[x0,x1]

	return data


# Create the date set
#data = createPCASet(100, 4.0)
data = createFahreneitCelciusSet(100, 4.0)

# Plot the data
plotDataSet(data)

# Scale and center the data
data_normalized = scale(data)

# Perform PCA
pca = PCA()
data_reduced = pca.fit_transform(data_normalized)

# PCA information
print "Number of principal components: "+str(pca.n_components_)
print "Components variance:"+str(pca.explained_variance_)
print "Components variance ratio:"+str(pca.explained_variance_ratio_)
print "Principal components: "+str(pca.components_)

# Manually reconstruct the original data set with using only the first principal component
data_manually_reconstructed = np.dot(data_reduced[:,0].reshape((100,1)),pca.components_[0].reshape((1,2)))
plotDataSet(data_manually_reconstructed)

# Reconstruct original data
# Perform PCA asking for one component
pca = PCA(1)
data_reduced = pca.fit_transform(data_normalized)
data_reconstructed = pca.inverse_transform(data_reduced)

# Plot data
plotDataSet(data_reconstructed)

print "Mean squared reconstruction error: "+str(np.mean(np.linalg.norm(data_normalized-data_reconstructed, axis=1)))
