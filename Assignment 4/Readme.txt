Name : Anchit Vidyut Navelkar
Roll No : 12CS30004

Requirements:

opencv, matplotlib, numpy, scipy

Running the Code:

python 12CS30004_Assignment3.py <filename>

For Part(a) and Part(b) the <filename> argument needs to be provided while executing the code.


Saved Files:

HISTS.npy stores the histograms of test images so that they need not be input again for testing.


Function Documentation :

def partA(filename, gamut, minw, maxw)
	
	Executes Part A, provides a filtering function for colours between a given range of wavelengths

def createGamut(xyz_v)

	Creates the Gamut of colours using the csv file given

def getHistogram(imgw)

	Returns Histogram of the input array

def getNormalisedHistogram(filename, gamut)

	Calculates histogram of a file on its wavelengths

def partB(filename, gamut)

	Executes Part B

def findcentroids(data, clusters, k)
	
	Finds centers of the clusters given data, cluster classes and the number of clusters

def kmeans(data, k)

	Calculates the K clusters with Euclidean distance as a distance measure

def partC(gamut, K)

	Executes Part C
	
def start(args)

	Starting function, serves as a menu.