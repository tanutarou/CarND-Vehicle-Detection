## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_not_car]: ./output_images/car_not_car.png
[hog]: ./output_images/hog.png
[window]: ./output_images/window.png
[heat1]: ./output_images/heat1.png
[heat2]: ./output_images/heat2.png
[heat3]: ./output_images/heat3.png
[res1]: ./output_images/res1.png
[res2]: ./output_images/res2.png
[res3]: ./output_images/res3.png
[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test3]: ./output_images/test3.png
[test4]: ./output_images/test4.png
[test5]: ./output_images/test5.png
[test6]: ./output_images/test6.png
[hidden_car]: ./output_images/hide_car.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the `solution.ipynb`.This code is dupulicate of lesson 23.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of two of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and evaluated them by test accuracy. I choiced `orientations=11`, `pixels_per_cell=(16, 16)`, `cell_per_block=(2, 2)` in YCrCb colorspace. My model with this parameter got nice test accuracy 0.9881.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step ins contained in the 7th code cell of the `solution.ipynb`.
I trained a linear SVM using feature which contains hog feature(all channel), color histogram features and spatial features. I used bins of histogram is 8 and `spatial_size=(8, 8)`.
And I choiced SVM's parameter `C=0.1`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for Hog sub-sampling window search is contained in the 8th code cell of the `solution.ipynb`. This code is dupulicate of lesson 23.
I used 2 cells to step instead of using overlap rate and 64 as sampling rate. It means overlap of 75%.

I used various combination of scales and search region. I decided it as below table. There are many scale cars in an image. So, trying many setting was very important.

[y_start, y_end]|scale|
----------------|-----|
[350, 530]|1.0|
[350, 545]|1.5|
[350, 560]|2.0|
[350, 575]|2.5|
[350, 590]|3.0|


The below image shows all search windows.

![alt text][window]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][test1]  
![alt text][test2]  
![alt text][test3]  
![alt text][test4]  
![alt text][test5]  
![alt text][test6]  

I searched various combination of scale and search region(y_start, y_end) and the combination of features to optimize the performance of my classifier. I thought using all feature gave the best accuracy for test data. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also used time-series information. I saved heatmap of 10 frame and averaged them. A car doesn't  move rapidly. So, averaged heatmap become more robust for detection errors.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are resulting bouding boxes and their corresponding heatmaps:

heatmap| resuling bouding boxes|
-------|-----------------------|
![alt text][heat1] | ![alt text][res1] |
![alt text][heat2] | ![alt text][res2] |
![alt text][heat3] | ![alt text][res3] |




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* My pipeline cannot detect a partially hidden car like below. I think the reason is that cars in training data don't hide like this. So, I need to collect more data or make hidden cars data. Making the data is easy by cropping car image.  

![alt text][hidden_car]

* Though my SVC got 0.9881 test accuracy, false positives are little too much. So, I used time-series information. But, it's not sufficient to cope this problem. I think `project_video.wmv` has different road freature from training dataset. If I collect more data and train it, my pipeline may be improved.
