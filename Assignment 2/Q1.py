import numpy as np 
import cv2
import sys
image=None
mod_image=None
points=[]
points2=[]
p_h=0
p_w=0

def draw_point(event,x,y,flags,param):
	global mod_image, points
	if event == cv2.EVENT_FLAG_LBUTTON:
		cv2.rectangle(mod_image,(x-2,y-2),(x+2,y+2),(0,255,120),-1)
		points.append((x,y))

def draw_point2(event,x,y,flags,param):
	global mod_image, points2
	if event == cv2.EVENT_FLAG_LBUTTON:
		cv2.rectangle(mod_image,(x-2,y-2),(x+2,y+2),(120,0,255),-1)
		points2.append((x,y))

def show_rectified():
	global points, image, points2
	h, w, c=image.shape
	points=np.float32(points)
	points2 = np.float32(points2)
	M = cv2.getPerspectiveTransform(points,points2)

	dst = cv2.warpPerspective(image,M,(p_w, p_h))
	cv2.imshow("ResultQ1", dst)
	cv2.waitKey()
	cv2.imwrite("ResultQ1_ajanta.jpg", dst)

def collectpoints():
	cv2.namedWindow("Original")
	cv2.setMouseCallback("Original",draw_point)

	while(1):
		cv2.imshow("Original",mod_image)
		if (cv2.waitKey(20) & 0xFF == 27) | len(points)==4:
			break
	cv2.destroyAllWindows()

def collectpoints2():
	global points2, p_h, p_w
	cv2.namedWindow("Original-Rect")
	cv2.setMouseCallback("Original-Rect",draw_point2)

	while(1):
		cv2.imshow("Original-Rect",mod_image)
		if (cv2.waitKey(20) & 0xFF == 27) | len(points2)==2:
			break

	pt=points2
	p_h=pt[1][1]-pt[0][1]
	p_w=pt[1][0]-pt[0][0]
	points2=[[pt[0][0],pt[0][1]],[pt[1][0],pt[0][1]],[pt[1][0],pt[1][1]],[pt[0][0],pt[1][1]]]

	cv2.destroyAllWindows()

def start(image_name):
	global image, points, mod_image
	image=cv2.imread(image_name)
	h, w, c=image.shape
	image=cv2.resize(image, (w/5, h/5))
	mod_image=image.copy()
	print "This program performs the affine rectification of a given image."
	print "In the window that will open, please select 4 points(2 sets of parallel lines) to perofrm the rectification."
	print "Press ENTER to continue..."
	raw_input()

	collectpoints()

	print "Now, please select 2 opposite corners of the rectangle to which image will be projected."
	print "Press ENTER to continue..."
	raw_input()

	collectpoints2()
	print "Thank you. The points entered are " + str(points)
	print "Press ENTER to start affine rectification and display result."
	# raw_input()

	show_rectified()

if __name__ == "__main__":
	start(sys.argv[1])
