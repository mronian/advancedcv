import cv2
import numpy as np
from matplotlib import pyplot as plt

image1 = cv2.imread("DSC_0244.JPG")
image2 = cv2.imread("DSC_0245.JPG")
image3 = cv2.imread("DSC_0246.JPG")
image4 = cv2.imread("DSC_0247.JPG")
Width=700
Height=700
N=np.array([[2.0/Width,0,-1],
			[0,2.0/Height,-1],  
			[0,0,1]])

image1=cv2.resize(image1, (Width, Height))
image2=cv2.resize(image2, (Width, Height))
image3=cv2.resize(image3, (Width, Height))
image4=cv2.resize(image4, (Width, Height))
cur_image=None
cur_points=[]
points1=np.array([])
points2=np.array([])
points3=np.array([])
points4=np.array([])

def draw_point(event,x,y,flags,param):
	global cur_image, cur_points
	if event == cv2.EVENT_FLAG_LBUTTON:
		cv2.rectangle(cur_image,(x-2,y-2),(x+2,y+2),(50,55,150),-1)
		cur_points.append((x,y))

def drawpoints():
	global cur_image, cur_points
	for p in cur_points:
		cv2.rectangle(cur_image,(p[0]-2,p[1]-2),(p[0]+2,p[1]+2),(50,55,150),-1)

def collectpoints():

	cv2.namedWindow("Original")
	cv2.setMouseCallback("Original",draw_point)

	while(1):
		cv2.imshow("Original", cur_image)
		if (cv2.waitKey(20) & 0xFF == 27) | len(cur_points)==8:
			break
	cv2.destroyAllWindows()

def get_coefficients(pt1, pt2):
	a = pt2[0]
	b = pt2[1]
	a_ = pt1[0]
	b_ = pt1[1]
	return [ a_*a, a_*b, a_, b_*a, b_*b, b_, a, b, 1]

def normalizePoint(pt):
	global N
	pt=np.array([pt[0], pt[1], 1])
	return np.dot(N, pt)[0:2]

def getFundamentalMatrix(pts1, pts2):
	A = []
	for i in range(len(pts1)):
		norm_1=normalizePoint(pts1[i])
		norm_2=normalizePoint(pts2[i])
		A.append(get_coefficients(norm_2, norm_1))
	A = np.array(A)
	U, s, V = np.linalg.svd(A, full_matrices=False)
	F = V[-1].reshape((3,3))
	U, s, V = np.linalg.svd(F, full_matrices=False)
	s[-1] = 0.0
	s_avg=(s[0]+s[1])/2.0
	s[0]=s_avg
	s[1]=s_avg
	S = np.diag(s)
	F = np.dot(U, np.dot(S, V))
	F=np.dot(N.T, np.dot(F, N))
	return F/F[2,2]

def drawLines(img1,img2,lines,pts1,pts2):
	r,c,d = img1.shape
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2

def getEpilines(pt, F):
	temp=np.ones((8,3))
	temp[:,:-1]=pt

	return np.dot(F, temp.T).T

def getEpipoles(F):
	epipoles={}
	U,s,V=np.linalg.svd(F, full_matrices=False)
	epipoles['left']=U[:,-1]/U[:,-1][2]
	epipoles['right']=V[-1]/V[-1][2]

	return epipoles

def closestDistanceBetweenLines(a0,a1,b0,b1):
    A = a1 - a0
    B = b1 - b0
    A1 = A / np.linalg.norm(A)
    B1 = B / np.linalg.norm(B)
    cross = np.cross(A1, B1)
    denom = np.linalg.norm(cross)**2
    if (denom == 0):
        d0 = np.dot(A1,(b0-a0))
        d = np.linalg.norm(((d0*A1)+a0)-b0)
        return None,None,d
    t = (b0 - a0)
    det0 = np.linalg.det([t, B1, cross])
    det1 = np.linalg.det([t, A1, cross])
    t0 = det0/denom
    t1 = det1/denom
    pA = a0 + (A1 * t0)
    pB = b0 + (B1 * t1)
    d = np.linalg.norm(pA-pB)
    return pA,pB,d

def get3DCoordinates(pts1, pts2, P1, P2):
	norm_1=np.ones((8,3))
	norm_1[:,:-1]=pts1
	norm_2=np.ones((8,3))
	norm_2[:,:-1]=pts2

	M1=P1[:,0:3]
	P41=P1[:,3]
	M2=P2[:,0:3]
	P42=P2[:,3]

	M1_inv=np.linalg.inv(M1)
	M2_inv=np.linalg.inv(M2)

	M1_inv_pts1=np.dot(M1_inv, norm_1.T)
	M2_inv_pts2=np.dot(M2_inv, norm_2.T)
	M1_inv_P41=-np.dot(M1_inv, P41)
	M2_inv_P42=-np.dot(M2_inv, P42)

	points3D=[]
	for i in range(pts1.shape[0]):
		pA, pB, d=closestDistanceBetweenLines(M1_inv_pts1[:,i], M1_inv_P41, M2_inv_pts2[:,i], M2_inv_P42)
		points3D.append((pA+pB)/2)

	return np.array(points3D)

def skew(a):
	return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def RQ(P):
	r, c = P.shape
	R = np.zeros((r,r))
	Q = np.zeros((r,c))
	
	Q,R = np.linalg.qr(np.flipud(P).T)
	R = np.flipud(R.T)
	Q = Q.T
	return R[:,::-1],Q[::-1,:]

def getCameraCalibrationMatrix(P1, P2):
	K1, K2 = P1[:,0:3], RQ(P2[:,0:3])[0]
	return K1, K2
	
def process(img1, img2, pts1, pts2, name):
	F=getFundamentalMatrix(pts1, pts2)
	
	print "Fundamental Matrix :"
	print "--------------------"
	print F
	print ""

	# Epilines for Image 2
	lines2 = getEpilines(pts1, F)
	# Epilines for Image 1
	lines1 = getEpilines(pts2, F.T)
	img5,img6 = drawLines(img1.copy(),img2.copy(),lines1,pts1,pts2)
	img3,img4 = drawLines(img2.copy(),img1.copy(),lines2,pts2,pts1)

	epipoles = getEpipoles(F)

	print "Left Epipole :"
	print "--------------------"
	print epipoles['left']
	print ""

	print "Right Epipole :"
	print "--------------------"
	print epipoles['right']
	print ""

	Te = skew(epipoles['left'])
	P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
	P2 = np.vstack((np.dot(Te,F.T).T,epipoles['left'])).T
	
	K1, K2=getCameraCalibrationMatrix(P1,P2)

	print "Left Camera Calibration Matrix :"
	print "---------------------------------"
	print K1
	print ""

	print "Right Camera Calibration Matrix :"
	print "---------------------------------"
	print K2
	print ""

	points3D=get3DCoordinates(pts1, pts2, P1, P2)

	print "3D coordinates of the selected points:"
	print "---------------------------------"
	for i in range(pts1.shape[0]):
		print str(pts1[i])+" and "+str(pts2[i])+"  --------->  "+str(points3D[i])

	img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
	img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
	plt.subplot(121),plt.imshow(img5)
	plt.subplot(122),plt.imshow(img3)
	# plt.show()
	# plt.savefig(name)

def get_keypoints():
	global cur_image, cur_points, image1, image2, image3, image4, points1, points2, points3, points4
	print "In the window that will open, please select 8 keypoints for image1"
	print "Press ENTER to continue..."
	raw_input()
	cur_image=image1.copy()
	collectpoints()
	points1=np.array(cur_points)
	cur_points=[]
	print "Thank you. The points entered are " + str(points1)

	print "In the window that will open, please select 8 keypoints for image2"
	print "Press ENTER to continue..."
	raw_input()
	cur_image=image2.copy()
	collectpoints()
	points2=np.array(cur_points)
	cur_points=[]
	print "Thank you. The points entered are " + str(points2)

	print "In the window that will open, please select 8 keypoints for image3"
	print "Press ENTER to continue..."
	raw_input()
	cur_image=image3.copy()
	collectpoints()
	points3=np.array(cur_points)
	cur_points=[]
	print "Thank you. The points entered are " + str(points3)
	
	print "In the window that will open, please select 8 keypoints for image4"
	print "Press ENTER to continue..."
	raw_input()
	cur_image=image4.copy()
	collectpoints()
	points4=np.array(cur_points)
	cur_points=[]
	print "Thank you. The points entered are " + str(points4)

def start():
	global points1, points2, points3, points4
	get_keypoints()
	KP_MAT=[points1, points2, points3, points4]
	np.save("Keypoints", KP_MAT)

	# global cur_image, cur_points, image1, image2, image3, image4, points1, points2, points3, points4
	# KP_MAT=np.load("Keypoints.npy")

	# points1=KP_MAT[0]
	# points2=KP_MAT[1]
	# points3=KP_MAT[2]
	# points4=KP_MAT[3]

	process(image1, image2, points1, points2, name="1_2-3.png")
	process(image1, image3, points1, points3, name="1_3-3.png")
	process(image1, image4, points1, points4, name="1_4-3.png")
	process(image2, image3, points2, points3, name="2_3-3.png")
	process(image2, image4, points2, points4, name="2_4-3.png")
	process(image3, image4, points3, points4, name="3_4-3.png")

	# for item in [[image1, points1], [image2, points2], [image3, points3], [image4, points4]]:
	# 	cur_image=item[0].copy()
	# 	cur_points=item[1]

	# 	drawpoints()
	# 	cv2.imshow("POINTS", cur_image)
	# 	cv2.waitKey()

if __name__ == '__main__':
	start()