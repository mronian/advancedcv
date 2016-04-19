import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import scipy.spatial

BGR2XYZ=[[0.0193339, 0.1191920, 0.9503041],
		[0.2126729, 0.7151522, 0.0721750],
		[0.4124564, 0.3575761, 0.1804375]] 


def partA(filename, gamut, minw, maxw):

	img=cv2.imread(filename)
	h,w,_=img.shape
	img=cv2.resize(img, (w/4, h/4)) 
	h,w,_=img.shape
	cv2.namedWindow("Result")

	imgXYZ=np.dot(img, BGR2XYZ)
	imgxyz=np.zeros(img.shape)
	imgXYZ=imgXYZ.astype(np.float)
	for i in range(h):
		for j in range(w):
			sumI=imgXYZ[i,j,0]+imgXYZ[i,j,1]+imgXYZ[i,j,2]
			if sumI==0.0:
				imgxyz[i,j]=[0.0,0.0,0.0]
				continue
			for k in range(3):
				imgxyz[i,j,k]=imgXYZ[i,j,k]/sumI

	img_angles=np.rad2deg(np.arctan2((imgxyz[:,:,1]-(1.0/3.0)),(imgxyz[:,:,0]-(1.0/3.0))))
	for i in range(h):
		for j in range(w):
			if img_angles[i,j]<0.0:
				img_angles[i,j]+=360.0

	res=img.copy()
	maxa=gamut[np.searchsorted(gamut[:,0], minw),1]
	mina=gamut[np.searchsorted(gamut[:,0], maxw),1]
	for i in range(h):
		for j in range(w):
			if img_angles[i,j]<mina or img_angles[i,j]>maxa:
				res[i,j]=[0,0,0]
	cv2.imshow('Result',res)
	cv2.waitKey()

def createGamut(xyz_v):
	angles=np.zeros((xyz_v.shape[0],2))
	angles[:,0]=xyz_v[:,0]
	angles[:,1]=np.rad2deg(np.arctan2((xyz_v[:,2]-(1.0/3.0)),(xyz_v[:,1]-(1.0/3.0))))
	for i in range(len(angles)):
		if angles[i,1]<0.0:
			angles[i,1]+=360.0
	return angles

def getHistogram(imgw):

	hist=np.zeros((831,1))
	for i in range(imgw.shape[0]):
		for j in range(imgw.shape[1]):
			hist[imgw[i,j]]+=1

	return hist/sum(hist)

def getNormalisedHistogram(filename, gamut):	
	img=cv2.imread(filename)
	h,w,_=img.shape
	img=cv2.resize(img, (w/4, h/4)) 
	h,w,_=img.shape
	imgXYZ=np.dot(img, BGR2XYZ)

	imgxyz=np.zeros(img.shape)
	imgXYZ=imgXYZ.astype(np.float)
	for i in range(h):
		for j in range(w):
			sumI=imgXYZ[i,j,0]+imgXYZ[i,j,1]+imgXYZ[i,j,2]
			if sumI==0.0:
				imgxyz[i,j]=[0.0,0.0,0.0]
				continue
			for k in range(3):
				imgxyz[i,j,k]=imgXYZ[i,j,k]/sumI

	img_angles=np.rad2deg(np.arctan2((imgxyz[:,:,1]-(1.0/3.0)),(imgxyz[:,:,0]-(1.0/3.0))))
	for i in range(h):
		for j in range(w):
			if img_angles[i,j]<0.0:
				img_angles[i,j]+=360.0

	gamut=gamut[gamut[:,1].argsort()]
	img_wavelengths=np.zeros(img_angles.shape)
	for i in range(h):
		for j in range(w):
			try:
				img_wavelengths[i,j]=gamut[np.searchsorted(gamut[:,1], img_angles[i,j]),0]
			except IndexError:
				img_wavelengths[i,j]=830

	hist=getHistogram(img_wavelengths)
	return hist

def partB(filename, gamut):	

	hist=getNormalisedHistogram(filename, gamut)

	plt.plot(hist)
	plt.show()

def findcentroids(data, clusters, k):
	result = np.empty(shape=(k,) + data.shape[1:])
	for i in range(k):
		np.mean(data[clusters == i], axis=0, out=result[i])
	return result

def kmeans(data, k):
	centres = data[np.random.choice(np.arange(len(data)), k, False)]
	for i in range(max(50, 1)):
		sqdists = scipy.spatial.distance.cdist(centres, data, 'sqeuclidean')
		clusters = np.argmin(sqdists, axis=0)
		new_centres = findcentroids(data, clusters, k)
		if np.array_equal(new_centres, centres):
			break
		centres = new_centres
	return clusters

def partC(gamut, K):

	hists=[]
	for i in os.listdir(os.getcwd()+"/Test_Images/"):
		print i
		hists.append(getNormalisedHistogram("Test_Images/"+str(i), gamut))
	hists=np.array(hists)
	np.save("HISTS", hists)
	# hists=np.load("HISTS.npy")
	hists=np.reshape(hists, (21, 831))

	classes=kmeans(hists, K)
	print classes
	return classes
	
def start(args):
	cie_XYZ = np.genfromtxt('ciexyz31_1.csv', delimiter=',')
	cie_xyz=np.zeros(cie_XYZ.shape)
	cie_xyz[:,0]=cie_XYZ[:,0]

	h,w=cie_xyz.shape

	for i in range(h):
		sumI=cie_XYZ[i,1]+cie_XYZ[i,2]+cie_XYZ[i,3]
		if sumI==0.0:
			cie_xyz[i,1:4]=[0.0,0.0,0.0]
			continue
		for k in range(1,4):
			cie_xyz[i,k]=cie_XYZ[i,k]/sumI

	gamut=createGamut(cie_xyz)
	
	print "This program does the following :"
	print "1. Provide an interface to filter colours based on wavelength [Filename given as argument]"
	print "2. Compute a normalised histogram of wavelengths [Filename given as argument]"
	print "3. Cluster the given set of images in Test_Images directory using kMeans algorithm with Euclidean Distance"
	print "Please Enter your choice:",
	choice=raw_input()
	if choice=="1":
		print "Please Enter Maximum Wavelength:",
		maxa=int(raw_input())
		print "Please Enter Minimum Wavelength:",
		mina=int(raw_input())
		partA(args[1], gamut, mina, maxa)
	elif choice=="2":
		partB(args[1], gamut)
	elif choice=="3":
		print "Please Enter K (No. of centers):",
		k=int(raw_input())
		partC(gamut, k)
	else :
		print "Wrong Choice. Program Exiting."

if __name__ == '__main__':
	start(sys.argv)