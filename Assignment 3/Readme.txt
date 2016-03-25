Name : Anchit Vidyut Navelkar
Roll No : 12CS30004

Requirements:

opencv, matplotlib, numpy

Running the Code:

python 12CS30004_Assignment3.py

In the GUI that pops up, select 8 keypoints for each of the 4 images one by one.
Press ENTER once keypoints for one image are selected to select for the next one. 

The corresponding keypoints selected in each image should match i.e should be the same point in 3D.

Once all keypoints are selected, each pair of images is operated upon, giving the Fundamental Matrix, Epilines, Epipoles, Camera Calibration Matrix, 3D Coordinates as output.



Saved Files:

Keypoints.npy stores the keypoints so that they need not be input again for testing.


Function Documentation :

def draw_point(event,x,y,flags,param)

	Used to draw markers for the points collected to calculate fundamental matrix

def collectpoints()

	Used to collect points for fundamental matrix using mouse click

def get_coefficients(pt1, pt2)

	Returns a row of the A matrix given 2 points

def skew(a)

	Calculates a matrix A such that a x v=Av for all v

def getEpipoles(F)
	
	Returns epipoles for a fundamental matrix

def getEpilines(pt, F)

	Returns epilines for a set of points and fundamental matrix

def drawLines(img1,img2,lines,pts1,pts2)

	Used to draw the epilines

def getFundamentalMatrix(pts1, pts2)

	Calculates fundamental matrix from the two sets of keypoints

def RQ(P)

	Performs the RQ decomposition of P 

def get3DCoordinates(pts1, pts2, P1, P2)

	Gives te 3D Coordinates of the 2 sets of points given both P matrices

def closestDistanceBetweenLines(a0,a1,b0,b1)

	Calculates the closest distance between two lines

def getCameraCalibrationMatrix(P1, P2)

	Calculates the camera calibration matrix

def normalizePoint(pt)
	
	Normalises a point for fundamental matrix calculation using SVD

def get_keypoints()
	
	Collects keypoints from GUI for all the 4 images

def process(img1, img2, pts1, pts2, name)
	
	Operates on pairs of images to calculate Fundamental Matrix, Epilines, Epipoles, Camera Calibration Matrix, 3D Coordinates

def start()
	
	Serves as main function
