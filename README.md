# Analysis and Data Extraction Tools
**This project provides a toolkit for loading, visualizing, analyzing, and processing MRI images in Python. It includes functions for:**
- Loading DICOM MRI images chosen by the user
- Displaying 2D slices and 3D volumes with customization like axes, colorbars etc.
- Extracting and printing metadata from the images
- Calculating metrics like histogram, center of mass, spatial extent
- Applying filters like Gaussian, median, edge detection
- Performing image processing operations like erosions, rotations, translations, resizing
- Comparing two images to find similarity metrics like mean absolute error and intersection over union
- Simple machine learning with scikit-learn models like SVM, KNN, regression
  
**The project relies on standard Python scientific computing packages:**
- numpy, scipy: for numerical processing of arrays
- matplotlib: for visualization and plotting
- scikit-learn: for machine learning models
- scikit-image: for image processing operations
- pandas: for general data analysis
- tkinter: for file dialogs to choose images

**The code is organized into functions for each operation which can be imported and used modularly as part of a pipeline.** **Examples are provided for calling each function.**

**To use the toolkit, simply import the necessary functions and apply them to your own MRI data. The functions provide** **reusable components for loading data, preprocessing, feature extraction, and machine learning when analyzing MRI images.**

**Some potential applications include:**
- Quality control on MRI scanner output
- Automated analysis like brain tumor detection
- Image regression tasks to predict clinical variables from scans
- Data preprocessing and feature engineering for ML models
- Medical imaging research to process and explore MRI images
