Name : Anchit Vidyut Navelkar
Roll No : 12CS30004

Running the Code [Choose option from the menu which appears]:

python Assignment1.py

Files Saved in Folder :

The program stores certain images and arrays to file after calculation for use by other functions to help increase speed. These files are :

	X-Gradient.npy : Sobel X-Gradient
	Y-Gradient.npy : Sobel Y-Gradient
	Gradient Magnitude.npy : Sobel Gradient Magnitude

	PSNR.npy : Stores PSNR vs p values
	PSNR_Opencv : Stores PSNR vs p values calculated using Opencv functions

	DCT_MATRIX.npy : DCT Coefficient Matrix
	iDCT_IMAGE.npy : iDCT Image

Function Documentation :

read_and_display()

	Reads the two images and displays them.

gray(image):
	
	Takes a colour image as input and return the grayscale image.

convert_to_grayscale():

	Reads the two images, converts to grayscale using gray() and displays them.

median_mean_filter(noisy_image):

	Takes noisy image as input and returns mean and median filtered images.

salt_and_pepper(image, p):
	
	Adds salt and pepper noise to an image with probability p, both given as inputs.

add_and_remove_salt_and_pepper():

	Reads "cap.bmp", adds salt and pepper noise, does median and mean filtering and displays the result.

get_psnr(original, new):
	
	Returns the PSNR value between the two images given as input.

psnr_p_plot():

	Does add_and_remove_salt_and_pepper() for different "p" values and plots the PSNR vs. P curve.

psnr_p_plot_opencv():
	
	Does add_and_remove_salt_and_pepper(), but with opencv functions for median and mean filter for different "p" values and plots the PSNR vs. P curve.

convolve(image, window):

	Takes an image and window, performs convolution and returns the resulting image.

sobel():
	
	Performs sobel filter operation on "lego.tif", saves to file the X-gradient, Y-Gradient and Gradient Magnitude images for use by other functions and displays them.

get_edges():

	Reads the Gradient Magnitude image saved by the sobel() function and allows thresholding using a trackbar
	to get the edges.

Cu(idx):
	
	Gives value of coefficient for DCT and iDCT conversion

dct(img_gray):

	Converts image to dct and saves the matrix to file.

idct(dct_matrix):

	Takes dct matrix and converts back to image.

dct_and_idct():

	Uses the dct() and idct() functions on cap.bmp and shows the result.

gaussian_kernel(size):

	Returns gaussian kernel of given size.

local(image, threshold):

	Finds local maxima above the threshold value and returns and binary matrix.

corner_harris():

	Performs corner harris on Lego.tif and saves the R matrix to file.

draw(image, r):

	Takes an image and r matrix and draws circle for detected corners.

nothing(x):

	Used as dummy function for trackbars.

find_corners():

	Uses saved R matrix to display corners on Lego.tif
