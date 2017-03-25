
## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog]: ./assets/hog.png
[heat]: ./assets/heatmap_test.png
[sliding]: ./assets/sliding_windows.png
[result]: http://img.youtube.com/vi/ydFYlFZM4M8/0.jpg

**Click on the image to go to the youtube video**

[![IMAGE ALT TEXT](http://img.youtube.com/vi/ydFYlFZM4M8/0.jpg)](https://youtu.be/ydFYlFZM4M8 "Project video")


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!
Also, the code for this project is stored in `P5-Vehicle-Detection.ipynb`

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


The code for this step is contained in the cell #6 of the IPython notebook. I started by loading the datasets and then I can extract the hog features by using the `skimage library` functions.

After some testing, I settled down on the following parameters for the feature extraction and training. I chose these parameters because they achieved the highest score in the training and the best results for the final algorithms of the project.
```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

Here you can see a couple examples on how do the hog features look like for a car and a non-car images:

![alt text][hog]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using around 17000 images total (almost uniformly splitted between car and non car images). I then extract features (such as color histogram and HOG) from them, and split those features into training and test data.
```
train_features size: (14208, 2580)
train_labels size: (14208,)
test_features size: (3552, 2580)
test_labels size: (3552,)
```
Then, I trained a `LinearSVC` to perform the classification task. I obtained an accuracy of 0.9755.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to perform a serch only on the lower half of the image. I also applied different size windows to account for the distance to the Ego-vehicle. This way I arrived to the following distribution of windows:
```
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[375, 475],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500],
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500],
                    xy_window=(144, 144), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[430, 550],
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[460, 580],
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
```
Resulting in this:
![alt text][sliding]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.
Once the model has been trained and the windows have been generated, I proceed to predict the content of the windows using the classifier. This generates hot windows that then are added together to form heat maps as the ones shown below.

![alt text][heat]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output/project_video_output.mp4)

The video can also be found in [youtube](https://youtu.be/ydFYlFZM4M8)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Once this is done, I take the resultant boxes and add "weight" to them and slightly enlarge them. Then, these frames are stored in a buffer, this way, I can make the system more likely to find cars in the vicinity of where they where in previous frames. The code for this section can be found in the `process_video` function

![result]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think that a larger dataset and/or image augmentation can help to improve the results of the algorithm and reduce false positives. Due to the nature of the problem, it would be interesting to try to solve it by using image segmentation and deep learning.

Since the datasets are divided into side, front and rear of the cars, we could implement different classifiers for each section and then combine them to create 3D bounding boxes.

To improve the tracking, we could implement a Kalman filter based solution that will also help with the false positives
