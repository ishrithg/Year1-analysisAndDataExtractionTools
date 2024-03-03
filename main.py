# Import Statements
import pandas as pd
import imageio
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skimage import data, color, exposure, morphology, measure
from skimage.filters import try_all_threshold, threshold_otsu, threshold_local, sobel, gaussian
from skimage.transform import rotate, rescale, resize
from skimage.restoration import inpaint, denoise_tv_chambolle, denoise_bilateral
from skimage.util import random_noise
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.feature import canny, corner_harris, corner_peaks, Cascade



# I have created functions that explain what the functions will do. You can call whichever function you like.
# I have created a function that is called All at the bottom that displays all of these functions.
# However, if you want detailed and bigger images, you might want to call the functions.

def Scan_Axis_On():

   """This function will display an image of a 2D MRI scan with the axes and the colorbar labeled.
   This requires you to choose a file from your computer. This is required be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   plt.imshow(im, cmap='gray')
   plt.plot()
   plt.axis()
   plt.colorbar()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Axis_On()

def IM_Data():

   """This function will print a metadata dictionary containing information regarding the patient and the MR scan.
   This requires you to choose a file from your computer. This is required be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   print('Metadata Dictionary: {}'.format(im.meta))

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# IM_Data()

def Data_Type():

   """This function will print the type of the MR scan.
   This requires you to choose a file from your computer. This is required be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   print('Data type: {}'.format(type(im)))

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Data_Type()

def Scan_Axis_Off():

   """This function will display an MR scan with the axes off.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   plt.imshow(im, cmap='gray')
   plt.plot()
   plt.axis('off')
   plt.colorbar()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Axis_Off()

def Volume_Dataset():

   """This function will display an image of a 3-DIMENSIONAL MRI scan. It will have all the slices of the image shown.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   volume = filedialog.askopenfilename()
   vol = imageio.volread(volume)
   index = vol.shape[0]
   fig, axes = plt.subplots(nrows=1, ncols=index)
   for image in range(index):
       im = vol[image, :, :]
       axes[image].imshow(im, cmap='gray')
       axes[image].axis('off')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Volume_Dataset()

def Scan_Info():

   """This function will print the datatype, size, and shape of the scan.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   print("Here is the datatype of the scan: {}".format(im.dtype))
   print("Here is the size of the scan: {}".format(im.size))
   print("Here is the shape of the scan: {}".format(im.shape))

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Info()

def Scan_Histogram():

   """This function will display a graph of a 2-dimensional MRI scan with the image intensities.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   sns.set()
   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   hist = ndi.histogram(im, min=0, max=255, bins=256)
   plt.plot(hist)
   plt.title('Histogram')
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Histogram()

def Scan_CDF():

   """This function will display the graph of a 2-dimensional MRI scan with the image intensities AND
   the graph of the CDF (Cumulative Distribution Function).
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   sns.set()
   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   hist = ndi.histogram(im, min=0, max=255, bins=256)
   cdf = hist.cumsum() / hist.sum()
   plt.plot(cdf)
   plt.title('CDF')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_CDF()

def Scan_Mask():

   """This function will display an image of a 2-dimensional MRI scan with a basic outline of the image.
   You might have to change the filter variable to get a better image of the basic outline.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   filter = threshold_otsu(im)
   mask = im > filter
   plt.imshow(mask, cmap='gray')
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Mask()

def Scan_Binary_Erosion():

   """This function will erode background pixels adjacent to the mask into mask pixels.
   You might have to change the value of the iterations to get a better and smoother image.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   erosion = ndi.binary_erosion(im, iterations=5)
   plt.imshow(erosion, cmap='gray')
   plt.axis('off')
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Binary_Erosion()

def Scan_Gaussian_Filter():

   """This function will display an image of a 2-dimensional MRI scan with a gaussian filter applied to the image.
   You might have to change the value of the sigma to get a better and smoother image.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   gaussian = ndi.gaussian_filter(im, sigma=2)
   plt.imshow(gaussian, cmap='gray')
   plt.axis('off')
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Gaussian_Filter()

def Scan_Median_Filter():

   """This function will display an image of a 2-dimensional MRI scan with a median filter applied to the image.
   You might have to change the value of the sigma to get a better and smoother image.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   median = ndi.median_filter(im, size=2)
   plt.imshow(median, cmap='gray')
   plt.axis('off')
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Median_Filter()

def Scan_Edges():

   """This function will display an image of a 2-dimensional MRI scan with the edges highlighted.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   edge1 = ndi.sobel(im, axis=0)
   edge2 = ndi.sobel(im, axis=1)
   edges = np.sqrt(np.square(edge1) + np.square(edge2))
   plt.imshow(edges, cmap='gray')
   plt.axis('off')
   plt.colorbar()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Edges()

def Scan_Parts():

   """This function will display an image of a 2-dimensional MRI scan with the parts of the image highlighted.
   You might have to change the numbers in the mask variable to get a better image of the parts highlighted.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   filtered = ndi.median_filter(im, size=3)
   mask1 = np.where(filtered > threshold_otsu(im), 1, 0)
   mask = ndi.binary_closing(mask1)
   labels, nlabels = ndi.label(mask)
   print('Number of labels: {}'.format(str(nlabels)))
   final_scan = np.where(mask, labels, np.nan)
   plt.imshow(final_scan, cmap='rainbow', alpha=0.75)
   plt.axis('on')
   plt.colorbar()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Parts()

def Scan_Spatial_Extent():

   """This function will calculate the spatial extent and the volume of the object.
   You might have to change the numbers in the mask variable to get a better image of the parts highlighted.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   volume = filedialog.askopenfilename()
   vol = imageio.volread(volume)
   d0, d1, d2 = vol.meta['sampling']
   dvoxel = d0 * d1 * d2
   filter = ndi.median_filter(vol, size=1)
   mask = np.where(filter > threshold_otsu(filter), 1, 0)
   parts = ndi.binary_closing(mask)
   labels, nlabels = ndi.label(parts)
   nvoxels = ndi.sum(1, labels, index=1)
   volume = nvoxels * dvoxel
   print('Spatial Extent: {}'.format(volume))

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Spatial_Extent()

def Scan_Center_of_Mass():

   """This function will display an image of a 2-dimensional MRI scan with the parts of the image highlighted.
   You might have to change the numbers in the mask variable to get a better sense of the center of mass.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   filter = ndi.median_filter(im, size=1)
   mask = np.where(filter > threshold_otsu(im), 1, 0)
   parts = ndi.binary_closing(mask)
   labels, nlabels = ndi.label(parts)
   com = ndi.center_of_mass(im, labels, index=1)
   print("Here is the center of mass for the scan: {}".format(com))

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Center_of_Mass()

def Scan_Translation():

   """This function can be utilized to translate and center an MR scan properly.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   com = ndi.center_of_mass(im)
   d0 = 128 - com[0]
   d1 = 128 - com[1]
   translate = ndi.shift(im, shift=[d0, d1])
   fig, axes = plt.subplots(nrows=1, ncols=2)
   axes[0].imshow(im, cmap='gray')
   axes[0].set_title('Original Image')
   axes[1].imshow(translate, cmap='gray')
   axes[1].set_title('Translated Image')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Translation()

def Scan_Rotation():

   """This function can be utilized to rotate an MR scan properly.
   You might have to change the angle of rotation in the rotate variable to adjust the image properly.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   rotated = ndi.rotate(im, angle=-25, axes=(0, 1), reshape=False)
   fig, axes = plt.subplots(nrows=1, ncols=2)
   axes[0].imshow(im, cmap='gray')
   axes[0].set_title('Original Image')
   axes[1].imshow(rotated, cmap='gray')
   axes[1].set_title('Rotated Image')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Rotation()

def Scan_Dilation():

   """This function can be utilized to translate and dilate an MR scan properly.
   You might have to change the values in the matrix variable to adjust the image properly.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   matrix = [[0.8, 0, -20],
             [0, 0.8, -10],
             [0, 0, 1]]
   dilate = ndi.affine_transform(im, matrix)
   com = ndi.center_of_mass(dilate)
   d0 = 128 - com[0]
   d1 = 128 - com[1]
   translate = ndi.shift(dilate, shift=[d0, d1])
   fig, axes = plt.subplots(nrows=1, ncols=2)
   axes[0].imshow(im, cmap='gray')
   axes[0].set_title('Original Image')
   axes[1].imshow(translate, cmap='gray')
   axes[1].set_title('Dilated Image')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Dilation()

def Scan_Zoom():

   """This function can be utilized to up-sample or down-sample an axis grid.
   You might have to change the values of the im_zoom variables to adjust the image properly.
   If the zoom factor is less than 1, it will shrink, and if the factor is greater than 1 it will grow.
   This requires you to choose a file from your computer. This is required to be a DICOM file."""

   root = tk.Tk()
   root.withdraw()
   image = filedialog.askopenfilename()
   im = imageio.imread(image)
   im_zoom = ndi.zoom(im, zoom=5, order=3)
   fig, axes = plt.subplots(nrows=1, ncols=2)
   axes[0].imshow(im, cmap='gray')
   axes[0].set_title('Original Image')
   axes[1].imshow(im_zoom, cmap='gray')
   axes[1].set_title('Zoomed Image')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Zoom()

def Scan_Mean_Absolute_Error():

   """This function can be utilized to find the Mean Absolute Error of 2 SIMILAR BUT DIFFERENT images from the SAME DATASET.
   This requires you to choose 2 files from your computer. They are required to be DICOM files."""

   root = tk.Tk()
   root.withdraw()
   image1 = filedialog.askopenfilename()
   im1 = imageio.imread(image1)
   root = tk.Tk()
   root.withdraw()
   image2 = filedialog.askopenfilename()
   im2 = imageio.imread(image2)
   err = im1 - im2
   abs_err = np.abs(err)
   mae = np.mean(abs_err)
   print("This is the mean absolute error value: {}".format(mae))
   fig, axes = plt.subplots(nrows=1, ncols=3)
   axes[0].imshow(im1, cmap='gray')
   axes[0].set_title('First Image')
   axes[1].imshow(im2, cmap='gray')
   axes[1].set_title('Second Image')
   axes[2].imshow(abs_err, cmap='gray')
   axes[2].set_title('MAE Image')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Mean_Absolute_Error()

def Scan_Intersection_of_Union():

   """This function can be utilized to find the Intersection of Union of 2 SIMILAR BUT DIFFERENT images from the SAME DATASET.
   This requires you to choose 2 files from your computer. They are required to be DICOM files."""

   root = tk.Tk()
   root.withdraw()
   image1 = filedialog.askopenfilename()
   im1 = imageio.imread(image1)
   root = tk.Tk()
   root.withdraw()
   image2 = filedialog.askopenfilename()
   im2 = imageio.imread(image2)
   mask1 = im1 > threshold_otsu(im1)
   mask2 = im2 > threshold_otsu(im2)
   intersection = mask1 & mask2
   union = mask1 | mask2
   iou = intersection.sum() / union.sum()
   print("This is the intersection of union value: {}".format(iou))
   fig, axes = plt.subplots(nrows=1, ncols=4)
   axes[0].imshow(im1, cmap='gray')
   axes[0].set_title('First Image')
   axes[1].imshow(im2, cmap='gray')
   axes[1].set_title('Second Image')
   axes[2].imshow(intersection, cmap='gray')
   axes[2].set_title('Intersection Image')
   axes[3].imshow(union, cmap='gray')
   axes[3].set_title('Union Image')
   plt.tight_layout()
   plt.show()

# The line of code below is an example of how to call the function. When you click run, you will have to select a file.
# Scan_Intersection_of_Union()