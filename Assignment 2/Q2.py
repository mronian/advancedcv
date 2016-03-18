import numpy as np 
import cv2

LOWE_RATIO=0.7
MIN_COUNT=4

def get_keypoints_and_features(image):

	descriptor = cv2.xfeatures2d.SIFT_create()
	kp, des = descriptor.detectAndCompute(image, None)

	return kp, des

def get_matched_keypoints(kp1, kp2, des1, des2):

	des_matcher=cv2.DescriptorMatcher_create("BruteForce")
	matches_all=des_matcher.knnMatch(des1, des2, 2)
	matches_good=[]

	for m,n in matches_all:
		if m.distance < LOWE_RATIO*n.distance:
			matches_good.append(m)

	return matches_good

def get_homography_matrix(kp1, kp2, matches):
	if len(matches)>MIN_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

		H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

		return H, status

def rectilinear_interpolation(image, times):

	h,w,c=image.shape
	output=np.zeros((h*times, w*times, c)).astype(np.uint8)

	for i in range(0,h*times,times):
		for j in range(0,w*times,times):
			output[i,j]=image[i/times, j/times]

	for i in range(1, h*times-1, times):
		for j in range(1,w*times-1,times):
			for k in range(0,times-1):
				for l in range(0,times-1):
					output[k+i,l+j]=output[i-1,j-1]*(-k)*(-l)+output[i-1, j-1+times]*(l+1)*(-k)+output[i-1+times, j]*(-l)*(k+1)+output[i-1+times, j-1+times]*(k+1)*(l+1)

	return output

def partA(image1, image2, image_blurred):

	print "Getting Keypoints using SIFT"

	keypoints1, descriptor1=get_keypoints_and_features(image1)
	keypoints2, descriptor2=get_keypoints_and_features(image2)
	keypoints_blurred, descriptor_blurred=get_keypoints_and_features(image_blurred)

	print "Got Keypoints and Descriptors"

	matches1 = get_matched_keypoints(keypoints1, keypoints_blurred, descriptor1, descriptor_blurred)
	matches2 = get_matched_keypoints(keypoints2, keypoints_blurred, descriptor2, descriptor_blurred)

	print "Got Good Matches"

	print "Getting Homography"

	H1, status1 = get_homography_matrix(keypoints1, keypoints_blurred, matches1)
	H2, status2 = get_homography_matrix(keypoints2, keypoints_blurred, matches2)

	print "Got Homography"

	h,w,c=image_blurred.shape

	img1_tr = cv2.warpPerspective(image1, H1, (w,h))
	img2_tr = cv2.warpPerspective(image2, H2, (w,h))

	img2gray = cv2.cvtColor(img1_tr,cv2.COLOR_BGR2GRAY)
	ret, mask1_inv = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)

	img2gray = cv2.cvtColor(img2_tr,cv2.COLOR_BGR2GRAY)
	ret, mask2_inv = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)

	dst = cv2.bitwise_and(image_blurred,image_blurred, mask = mask1_inv)
	dst = cv2.add(dst, img1_tr)

	dst = cv2.bitwise_and(dst,dst, mask = mask2_inv)
	dst = cv2.add(dst, img2_tr)

	cv2.imshow("Resulta", dst)
	cv2.waitKey()
	cv2.imwrite("Resulta.jpg", dst)

def partB(image1, image2):

	print "Getting Homography between Image2 and Image1"

	keypoints1, descriptor1=get_keypoints_and_features(image1)
	keypoints2, descriptor2=get_keypoints_and_features(image2)

	matches = get_matched_keypoints(keypoints2, keypoints1, descriptor2, descriptor1)

	H, status = get_homography_matrix(keypoints2, keypoints1, matches)

	h1,w1,c1=image1.shape
	h2,w2,c2=image2.shape

	print "Getting Projection of Image 1 using Homography I1=H(I2)"

	output = cv2.warpPerspective(image2, H, ( w1+w2, h2))

	rows,cols,channels = image1.shape

	print "Using mask of Image2 to compute panorama"

	roi = output[0:rows,0:cols]
	img2gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
	fg = cv2.bitwise_and(image1,image1,mask = mask)
	dst = cv2.add(bg,fg)
	output[0:rows,0:cols] = dst

	cv2.imshow("Resultb", output)
	cv2.waitKey()
	cv2.imwrite("Resultb.jpg", output)

def partC(image1, image2, choice):

	h,w,c=image1.shape
	times=2
	print "Getting Homography between Image1 and Image2"

	keypoints1, descriptor1=get_keypoints_and_features(image1)
	keypoints2, descriptor2=get_keypoints_and_features(image2)

	matches = get_matched_keypoints(keypoints2, keypoints1, descriptor2, descriptor1)

	H, status = get_homography_matrix(keypoints2, keypoints1, matches)

	proj2on1 = cv2.warpPerspective(image2, H, ( w, h))
	output=rectilinear_interpolation(image1, times)
	
	if choice==2:
		proj_int=rectilinear_interpolation(proj2on1, times)

	o_h,o_w,o_c=output.shape

	for i in range(o_h):
		for j in range(o_w):

			if choice==1:
				output[i,j][0]=max(output[i,j][0], proj2on1[i/times, j/times][0])
				output[i,j][1]=max(output[i,j][1], proj2on1[i/times, j/times][1])
				output[i,j][2]=max(output[i,j][2], proj2on1[i/times, j/times][2])
			elif choice==2:
				output[i,j][0]=max(output[i,j][0], proj_int[i, j][0])
				output[i,j][1]=max(output[i,j][1], proj_int[i, j][1])
				output[i,j][2]=max(output[i,j][2], proj_int[i, j][2])

	cv2.imshow("Resultc", output)
	cv2.waitKey()
	if choice==1:
		cv2.imwrite("Resultc.jpg", output)
	elif choice==2:
		cv2.imwrite("Resultc-2.jpg", output)
		
def start():

	image1=cv2.imread("Ajanta_1.jpg")
	image2=cv2.imread("Ajanta_2.jpg")
	image_blurred=cv2.imread("Ajanta_blurred.jpg")
	image_blurred = cv2.resize(image_blurred, (0,0), fx=0.25, fy=0.25)
	image1 = cv2.resize(image1, (0,0), fx=0.25, fy=0.25)
	image2 = cv2.resize(image2, (0,0), fx=0.25, fy=0.25)

	print "This program solves Q2 of the assignment."
	print "Which part do you want to see?"
	print "1. Restore blurred image using homographies computed from two clear images"
	print "2. Mosaic 2 images"
	print "3. Upscale one image using the other"
	print "Enter your choice [0 to exit]:",

	choice=int(raw_input())

	if choice==1:
		partA(image1, image2, image_blurred)
	elif choice==2:
		partB(image1, image2)
	elif choice==3:
		print "1. Use method 1"
		print "2. Use method 2"
		print "Check the report file for a description of each method."
		print "Enter your choice:",
		ch=int(raw_input())
		partC(image1, image2, ch)

if __name__ == "__main__":
	start()