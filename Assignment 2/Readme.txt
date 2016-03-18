Name : Anchit Vidyut Navelkar
Roll No : 12CS30004

Question 1:

Running the Code:

python Q1.py <filename>

In the GUI that pops up, select 4 points in clockwise direction for calculating homography.
Press ENTER once all points are selected.

Now, select 2 opposite corners of the rectangle to project the image to.
Press ENTER once done.

Result will be displayed in a new window.

Function Documentation :

def draw_point(event,x,y,flags,param)

	Used to draw markers for the points collected to calculate homography

def draw_point2(event,x,y,flags,param)
	
	Used to draw markers for the points collected to project image to

def show_rectified()

	Used to find the correted image

def collectpoints()

	Used to collect points for homography using mouse click

def collectpoints2()

	Used to collect points for projection using mouse click

def start(image_name)
	
	Serves as main function


Question 2:

Running the Code [Choose option from the menu which appears]:

python Q2.py

Function Documentation :

def get_keypoints_and_features(image):

	Returns SIFT keypoints and descriptors

def get_matched_keypoints(kp1, kp2, des1, des2):

	Returns good matches of keypoints thresholded by LOWE's ratio

def get_homography_matrix(kp1, kp2, matches)
	
	Returns homography matrix between two sets of keypoints and good matches

def rectilinear_interpolation(image, times)

	Finds bilinear interpolation of image upscaled by x(times)

def partA(image1, image2, image_blurred)

	Restores a blurred image using 2 other clear images of the same scene

def partB(image1, image2)

	Creates a mosaic of 2 given images

def partC(image1, image2)

	Upscales one image using information from other image with bilinear interpolation
		
def start()

	Serves as main function