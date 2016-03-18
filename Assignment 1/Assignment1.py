import numpy as np 
import cv2
import math
import time
import matplotlib.pyplot as plt

def read_and_display():
	image1=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
	image2=cv2.imread("cap.bmp", cv2.IMREAD_UNCHANGED)

	cv2.imshow("Lego.tif", image1)
	cv2.imshow("Cap.bmp", image2)

	cv2.waitKey()

def gray(image):
	r,g,b=image[:,:,0],image[:,:,1],image[:,:,2]
	image_gray=0.299*r+0.587*g+0.114*b

	return image_gray.astype(np.uint8)

def convert_to_grayscale():
	image1=cv2.imread("lego.tif")
	image2=cv2.imread("cap.bmp")

	image1_gray=gray(image1)
	image2_gray=gray(image2)

	cv2.imshow("Lego.tif", image1_gray)
	cv2.imshow("Cap.bmp", image2_gray)

	cv2.waitKey()

def median_mean_filter(noisy_image):
	WINDOWSIZE=1
	#Median & Mean Filters	
	median_filtered_image=noisy_image.copy()
	mean_filtered_image=noisy_image.copy()
	for i in range(WINDOWSIZE, noisy_image.shape[0]-WINDOWSIZE):
		for j in range(WINDOWSIZE, noisy_image.shape[1]-WINDOWSIZE):
			window=noisy_image[i-WINDOWSIZE:i+WINDOWSIZE, j-WINDOWSIZE:j+WINDOWSIZE]
			median_filtered_image[i,j]=np.median(window)
			mean_filtered_image[i,j]=np.mean(window)
	return median_filtered_image, mean_filtered_image

def salt_and_pepper(image, p):
	noisy_image=image.copy()
	#Adding Salt
	for i in range(noisy_image.shape[0]):
		for j in range(noisy_image.shape[1]):
			randum_num=np.random.random_sample()
			if randum_num<p/2:
				noisy_image[i,j]=0

	#Adding Pepper
	for i in range(noisy_image.shape[0]):
		for j in range(noisy_image.shape[1]):
			randum_num=np.random.random_sample()
			if randum_num<p/2:
				noisy_image[i,j]=255

	return noisy_image

def add_and_remove_salt_and_pepper():

	image=cv2.imread("cap.bmp", cv2.IMREAD_UNCHANGED)
	noisy_image=image.copy()
	print "Enter the probability p used for adding noise:",
	p=float(raw_input())

	if p>1:
		print "Probability should be less than 1. Program exiting......."
	else :

		noisy_image=salt_and_pepper(image, p)
		median_filtered_image, mean_filtered_image=median_mean_filter(noisy_image)

		cv2.imshow("Original Image", image)
		cv2.imshow("Noisy Image", noisy_image)
		cv2.imshow("Median Filtered Image", median_filtered_image)
		cv2.imshow("Mean Filtered Image", mean_filtered_image)
		cv2.waitKey()

def get_psnr(original, new):
	original=original.astype(int)
	new=new.astype(int)
	mse=0.0
	for i in range(original.shape[0]):
		for j in range(original.shape[1]):
			mse=mse+(original[i,j]-new[i,j])**2

	mse=mse/((original.shape[0])*(original.shape[1]))
	psnr=20*np.log10(255)-10*np.log10(mse)

	return psnr

def psnr_p_plot():
	image=cv2.imread("cap.bmp", cv2.IMREAD_UNCHANGED)
	
	plot_array=[]

	for p in np.arange(0.1,0.2,0.01):
		print p
		noisy_image=salt_and_pepper(image, p)
		median_filtered_image, mean_filtered_image=median_mean_filter(noisy_image)
		psnr_median=get_psnr(image, median_filtered_image)
		psnr_mean=get_psnr(image, mean_filtered_image)

		plot_array.append((p,psnr_median, psnr_mean))

	plot_array=np.array(plot_array)
	np.save("PSNR", plot_array)
	plt.plot(plot_array[:, 0], plot_array[:, 1], 'ro')
	plt.plot(plot_array[:, 0], plot_array[:, 2], 'bo')
	plt.axis([0.09, 0.21, 15,35])
	plt.show()
	print plot_array

def psnr_p_plot_opencv():
	image=cv2.imread("cap.bmp", cv2.IMREAD_UNCHANGED)
	
	plot_array=[]

	for p in np.arange(0.1,0.2,0.01):
		print p
		noisy_image=salt_and_pepper(image, p)
		median_filtered_image=cv2.medianBlur(noisy_image, 3)
		mean_filtered_image=cv2.blur(noisy_image, (3,3))
		psnr_median=get_psnr(image, median_filtered_image)
		psnr_mean=get_psnr(image, mean_filtered_image)

		plot_array.append((p,psnr_median, psnr_mean))

	plot_array=np.array(plot_array)
	np.save("PSNR_Opencv", plot_array)
	plt.plot(plot_array[:, 0], plot_array[:, 1], 'rs')
	plt.plot(plot_array[:, 0], plot_array[:, 2], 'bs')
	plt.axis([0.09, 0.21, 15,35])
	plt.show()
	print plot_array

def convolve(image, window):
	tmp_a = np.zeros((
	    image.shape[0] + window.shape[0] / 2 * 2,
	    image.shape[1] + window.shape[1] / 2 * 2
	), float)
	tmp_a[
	window.shape[0] / 2:window.shape[0] / 2 + image.shape[0],
	window.shape[1] / 2:window.shape[1] / 2 + image.shape[1]
	] = image
	result = np.zeros(image.shape, float)
	for x in range(result.shape[0]):
		for y in range(result.shape[1]):
			result[x, y] = np.sum(
				tmp_a[x:x + window.shape[0], y:y + window.shape[1]] * window
			)
	return result

def sobel():
	image=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
	grayscale=gray(image)

	x_grad=grayscale.copy().astype(float)
	y_grad=grayscale.copy().astype(float)
	gradient_mag=grayscale.copy().astype(float)
	edges=grayscale.copy().astype(float)

	sumx=0
	sumy=0
	sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	sobely=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

	x_grad=convolve(grayscale, sobelx)
	y_grad=convolve(grayscale, sobely)
	gradient_mag=(x_grad**2+y_grad**2)**0.5

	np.save("X-Gradient.npy", x_grad)
	np.save("Y-Gradient.npy", y_grad)
	
	x_grad=(x_grad-np.amin(x_grad))*255/(np.amax(x_grad)-np.amin(x_grad))
	y_grad=(y_grad-np.amin(y_grad))*255/(np.amax(y_grad)-np.amin(y_grad))
	gradient_mag=(gradient_mag-np.amin(gradient_mag))*255/(np.amax(gradient_mag)-np.amin(gradient_mag))
	
	np.save("Gradient Magnitude.npy", gradient_mag)

	x_grad=x_grad.astype(np.uint8)
	y_grad=y_grad.astype(np.uint8)
	gradient_mag=gradient_mag.astype(np.uint8)
	
	cv2.imshow("X-Gradient", x_grad)
	cv2.imshow("Y-Gradient", y_grad)
	cv2.imshow("Gradient Magnitude", gradient_mag)
	cv2.waitKey()

def get_edges():
	gradient_mag=np.load("Gradient Magnitude.npy")
	cv2.namedWindow("Edges")
	cv2.createTrackbar("Threshold", "Edges", 0, 255, nothing)
	c=0
	while c!=27:
		thresh=cv2.getTrackbarPos("Threshold", "Edges")
		# print thresh
		thresholded=np.greater(gradient_mag, thresh).astype(np.uint8)*255
		# print thresholded
		cv2.imshow("Edges", thresholded)
		c=cv2.waitKey(33)
		if c==27:
			break

def Cu(idx):
	if idx==0:
		return 1/(2*math.sqrt(2))
	return (1.0/2)

def dct(img_gray):
	img_gray = img_gray.astype(float)
	height, width = img_gray.shape
	N = 8
	Cosines = np.zeros((N,N),float)
	DCT = np.zeros((height,width),float)
	for i in range(N):
		for j in range(N):
			Cosines[i][j] = math.cos((2*i + 1)*j*math.pi/(2*N))
	i = 0
	j = 0
	cnt = 0
	while i <= height - N:
		j = 0
		print i
		while j <= width - N:
			for si in range(i,i+N):
				for sj in range(j,j+N):
					temp = 0.0
					for x in range(i,i+N):
						for y in range(j,j+N):
							temp += Cosines[x - i][si - i] * Cosines[y - j][sj - j] * img_gray[x][y]
					temp *= Cu(si-i) * Cu(sj-j)
					DCT[si][sj] = temp
			j += N
		i += N
	np.save("DCT_MATRIX", DCT)
	return DCT

def idct(dct_matrix):
	height, width = dct_matrix.shape
	N = 8
	Cosines = np.zeros((N,N),float)
	iDCT = np.zeros((height,width),float)
	for i in range(N):
		for j in range(N):
			Cosines[i][j] = math.cos((2*i + 1)*j*math.pi/(2*N))
	i = 0
	j = 0
	cnt = 0
	while i <= height - N:
		j = 0
		print i
		while j <= width - N:
			for si in range(i,i+N):
				for sj in range(j,j+N):
					temp = 0.0
					for x in range(i,i+N):
						for y in range(j,j+N):
							temp += Cu(x-i) * Cu(y-j)*Cosines[si-i][x-i] * Cosines[sj - j][y - j] * dct_matrix[x][y]
					iDCT[si][sj] = temp
			j += N
		i += N
	np.save("iDCT_IMAGE", iDCT)
	return iDCT

def dct_and_idct():
	image=cv2.imread("cap.bmp", cv2.IMREAD_UNCHANGED)
	# dct_matrix=dct(image)
	dct_matrix=np.load("DCT_MATRIX.npy")
	# print dct_matrix
	dct_matrix_norm=(dct_matrix-np.amin(dct_matrix))/(np.amax(dct_matrix)-np.amin(dct_matrix))
	# cv2.imshow("DCT", dct_matrix_norm)
	print "Doing IDCT"
	# idct_image=idct(dct_matrix)
	idct_image=np.load("iDCT_IMAGE.npy")
	# print idct_image
	idct_image_norm=(idct_image-np.amin(idct_image))/(np.amax(idct_image)-np.amin(idct_image))
	idct_image_norm*=255
	idct_image_norm=idct_image_norm.astype(np.uint8)
	# cv2.imwrite("DCT.jpg", dct_matrix)
	# cv2.imwrite("IDCT.jpg", idct_image)
	max_diff=0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i,j]!=idct_image_norm[i,j]:
				diff=image[i,j]-idct_image_norm[i,j]
				if diff>max_diff:
					max_diff=diff
	print max_diff
	dct_matrix_norm*=255
	cv2.imwrite("DCTNR.jpg", dct_matrix_norm)
	cv2.imwrite("IDCTNR.jpg", idct_image_norm)
	# cv2.imwrite("IDCT255.jpg", idct_image255)
	cv2.imshow("IDCT", idct_image)
	cv2.imshow("IDCTNR", idct_image_norm)
	cv2.waitKey()

def gaussian_kernel(size):
    size = int(size)
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size)))
    return g / g.sum()

def gaussian_sum(window):
	kernel=gaussian_kernel(13)

def local(m, threshold=0):
    r = np.zeros(m.shape)
    for x in range(1, r.shape[0] - 1):
        for y in range(1, r.shape[1] - 1):
            if m[x, y] > threshold and \
                            m[x, y] > m[x - 1, y - 1] and m[x, y] > m[x, y - 1] and m[x, y] > m[x + 1, y - 1] and \
                            m[x, y] > m[x - 1, y] and m[x, y] > m[x + 1, y] and \
                            m[x, y] > m[x - 1, y + 1] and m[x, y] > m[x, y + 1] and m[x, y] > m[x + 1, y + 1]:
                r[x, y] = 1
            else:
                r[x, y] = 0
    return r

def corner_harris():
	image=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
	image=gray(image)
	sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	sobely=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

	# Ix=convolve(image, sobelx)
	# Iy=convolve(image, sobely)
	
	Ix=np.load("X-Gradient.npy")
	Iy=np.load("Y-Gradient.npy")
	window = np.array([
		[0.0029690, 0.0133062, 0.0219382, 0.0133062, 0.0029690],
		[0.0133062, 0.0596343, 0.0983203, 0.0596343, 0.0133062],
		[0.0219382, 0.0983203, 0.1621028, 0.0983203, 0.0219382],
		[0.0133062, 0.0596343, 0.0983203, 0.0596343, 0.0133062],
		[0.0029690, 0.0133062, 0.0219382, 0.0133062, 0.0029690]
	])
	Ixx = Ix**2
	Ixy = Iy*Ix
	Iyy = Iy**2
	Sxx = convolve(Ixx, window)
	Syy = convolve(Iyy, window)
	Sxy = convolve(Ixy, window)

	r = Sxx * Syy - Sxy**2 - 0.06*(Sxx+Syy)**2

	np.save("R.npy", r)
	return r

def draw(image, r):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x in range(r.shape[0]):
        for y in range(r.shape[1]):
            if r[x, y] > 0.5:
                cv2.circle(result, (y, x), 5, (0, 0, 255))
    return result

def find_corners():
	image=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
	image=gray(image)
	# r=corner_harris()
	# print time.now()
	r=np.load("R.npy")
	c=0
	# print time
	r = local(r, np.amax(r)/500)
	corners = draw(image, r)
	cv2.imshow("Corners", corners)
	# print time.now()
	cv2.waitKey()

def nothing(x):
	pass
	
def menu():
	print "-------------------------------------------------------------------"
	print "Image Processing Primer"
	print "-------------------------------------------------------------------"
	print "Choose an option:"
	print "1. Read and display both images"
	print "2. Convert both images to grayscale and display"
	print "3. Add salt and pepper noise and perform median/mean filtering"
	print "4. Plot PSNR vs. P for median and mean filters"
	print "5. Sobel on lego.tif"
	print "6. Get Edges on lego.tif and Threshold (First run 5 to get sobel gradients and save to file)"
	print "7. DCT and iDCT on cap.bmp"
	print "8. Corner Harris on lego.tif"
	print "9. Find Corners with Thresholding on lego.tif (First run 8 to get R matrix and save to file)"
	print "10. Grayscale with standard library function and compare"
	print "11. Median/Mean filtering with standard library function and compare"
	print "12. Plot PSNR vs. P for median and mean filters with opencv functions"
	print "13. PSNR vs. P plot with standard library function and compare"
	print "14. Sobel Edges with standard library function and compare"
	print "15. DCT and iDCT with standard library function and compare"
	print "16. Corner Harris with standard library function and compare"
	print "-------------------------------------------------------------------"
	print("Enter your choice:"),
	choice=raw_input()
	font = cv2.FONT_HERSHEY_SIMPLEX

	if choice=="1":
		read_and_display()
	elif choice=="2":
		convert_to_grayscale()
	elif choice=="3":
		add_and_remove_salt_and_pepper()
	elif choice=="4":
		psnr_p_plot()
	elif choice=="5":
		sobel()
	elif choice=="6":
		get_edges()
	elif choice=="7":
		dct_and_idct()
	elif choice=="8":
		corner_harris()
	elif choice=="9":
		find_corners()
	elif choice=="10":
		image1=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
		image2=cv2.imread("cap.bmp")
		image1_gray=gray(image1)
		image2_gray=gray(image2)

		image1_gray_opencv=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
		image2_gray_opencv=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

		print "PSNR For Lego.tif : "+str(get_psnr(image1_gray, image1_gray_opencv))
		print "PSNR For cap.bmp : "+str(get_psnr(image2_gray, image2_gray_opencv))

		cv2.imshow("Lego.tif", image1_gray)
		cv2.imshow("Cap.bmp", image2_gray)
		cv2.imshow("Lego.tif - OpenCV", image1_gray_opencv)
		cv2.imshow("Cap.bmp - OpenCV", image2_gray_opencv)
		cv2.waitKey()

	elif choice=="11":
		image=cv2.imread("cap.bmp")
		image=gray(image)
		h, w=image.shape
		noisy_image=image.copy()
		print "Enter the probability p used for adding noise:",
		p=float(raw_input())

		if p>1:
			print "Probability should be less than 1. Program exiting......."
		else :

			noisy_image=salt_and_pepper(image, p)
			median_filtered_image, mean_filtered_image=median_mean_filter(noisy_image)
			median_filtered_image_opencv=cv2.medianBlur(noisy_image, 3)
			mean_filtered_image_opencv=cv2.blur(noisy_image, (3,3))
			# print "PSNR for Median Filter : "+str(get_psnr(median_filtered_image, median_filtered_image_opencv))
			# print "PSNR for Mean Filter : "+str(get_psnr(mean_filtered_image, mean_filtered_image_opencv)) 
			cv2.putText(image,'Original',(10,500), font, 2,255,2)
			cv2.putText(noisy_image,'Original with Salt and Pepper Noise',(10,500), font, 2,255,2)
			cv2.putText(median_filtered_image,'Median Filter',(10,500), font, 2,255,2)
			cv2.putText(mean_filtered_image,'Mean Filter',(10,500), font, 2,255,2)
			cv2.putText(median_filtered_image_opencv,'Median Filter Opencv',(10,500), font, 2,255,2)
			cv2.putText(mean_filtered_image_opencv,'Mean Filter Opencv',(10,500), font, 2,255,2)

			merged=np.zeros((h*2, w*3), np.uint8)
			merged[0:h, 0:w]=image
			merged[h:2*h, 0:w]=noisy_image
			merged[0:h, w:2*w]=median_filtered_image
			merged[h:2*h, w:2*w]=mean_filtered_image
			merged[0:h, 2*w:3*w]=median_filtered_image_opencv
			merged[h:2*h, 2*w:3*w]=mean_filtered_image_opencv
			merged=cv2.resize(merged, (w*3/2, h))
			cv2.imshow("Output", merged)
			cv2.waitKey()

	elif choice=="12":
		psnr_p_plot_opencv()

	elif choice=="13":
		psnr_p=np.load("PSNR.npy")
		psnr_p_opencv=np.load("PSNR_Opencv.npy")

		plt.plot(psnr_p[:, 0], psnr_p[:, 1], 'ro')
		plt.plot(psnr_p[:, 0], psnr_p[:, 2], 'bo')
		plt.plot(psnr_p_opencv[:, 0], psnr_p_opencv[:, 1], 'rs')
		plt.plot(psnr_p_opencv[:, 0], psnr_p_opencv[:, 2], 'bs')
		plt.axis([0.09, 0.21, 15,35])
		plt.show()

	elif choice=="14":
		image=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
		image=gray(image)
		gradient_mag=np.load("Gradient Magnitude.npy")
		sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
		sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
		gradient_mag_opencv=(sobelx**2 + sobely**2)**0.5
		h, w=gradient_mag.shape
		cv2.namedWindow("Edges")
		cv2.createTrackbar("Threshold", "Edges", 0, 255, nothing)
		merged=np.zeros((h, 2*w))
		c=0
		while c!=27:
			thresh=cv2.getTrackbarPos("Threshold", "Edges")
			edges=np.greater(gradient_mag, thresh).astype(np.uint8)*255
			edges_opencv=np.greater(gradient_mag_opencv, thresh).astype(np.uint8)*255
			cv2.putText(edges,'Edges',(10,500), font, 2, 255,2)
			cv2.putText(edges_opencv,'Edges Opencv',(10,500), font, 2, 255,2)
			merged[:, 0:w]=edges
			merged[:, w:2*w]=edges_opencv
			cv2.imshow("Edges", cv2.resize(merged, (w, h/2)))
			c=cv2.waitKey(33)
			if c==27:
				break

	elif choice=="15":
		image=cv2.imread("cap.bmp", cv2.IMREAD_UNCHANGED)
		height, width=image.shape
		N=8
		idct_image=np.load("iDCT_IMAGE.npy")
		idct_image_norm=(idct_image-np.amin(idct_image))/(np.amax(idct_image)-np.amin(idct_image))
		idct_image_norm*=255
		idct_image_norm=idct_image_norm.astype(np.uint8)
		imf = np.float32(image)/255.0
		dct_opencv = imf.copy()
		idct_opencv = imf.copy()
		# dct_opencv=cv2.dct(imf)
		# idct_opencv=cv2.idct(dct_opencv)
		i=0
		while i <= height - N:
			j = 0
			while j <= width - N:
				dct_opencv[i:i+N, j:j+N]=cv2.dct(imf[i:i+N, j:j+N])
				j += N
			i += N
		print dct_opencv.dtype
		
		i=0
		while i <= height - N:
			j = 0
			while j <= width - N:
				idct_opencv[i:i+N, j:j+N]=cv2.idct(dct_opencv[i:i+N, j:j+N])
				j += N
			i += N

		idct_opencv = (idct_opencv-np.amin(idct_opencv))*255/(np.amax(idct_opencv)-np.amin(idct_opencv))

		idct_opencv = idct_opencv.astype(np.uint8)

		cv2.imshow("IDCT", idct_image_norm)
		cv2.imshow("IDCT Opencv", idct_opencv)
		cv2.waitKey()

	elif choice=="16":
		image=cv2.imread("lego.tif", cv2.IMREAD_UNCHANGED)
		image=gray(image)
		h,w=image.shape
		r_opencv=np.zeros(image.shape)
		r=np.load("R.npy")
		c=0
		r = local(r, np.amax(r)/500)
		corners = draw(image, r)

		grayscale = np.float32(image)
		corners_opencv = cv2.cornerHarris(grayscale,2,3,0.06)

		corners_opencv = cv2.dilate(corners_opencv,None)

		r_opencv[corners_opencv>0.01*corners_opencv.max()]=1
		corners_opencv=draw(image, r_opencv)

		cv2.putText(corners,'Corners',(10,500), font, 2, (255,0,0),2)
		cv2.putText(corners_opencv,'Corners Opencv',(10,500), font, 2, (255,0,0),2)
		merged=np.zeros((h,2*w,3))
		merged[0:h, 0:w, :]=corners
		merged[0:h, w:2*w, :]=corners_opencv
		merged=cv2.resize(merged, (w, h/2)).astype(np.uint8)
		cv2.imshow("Corners", merged)
		cv2.waitKey()

	else :
		print "Wrong Input. Program Exiting...."
	
if __name__ == "__main__":
	menu()