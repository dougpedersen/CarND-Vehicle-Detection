##Project 5 Writeup  
###Vehicle Detection and Tracking

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
[image1]: ./output_images/vehicle1.png "Vehicle"
[image1a]: ./output_images/non-vehicle1.png "Not a vehicle"
[image4]: ./output_images/test_processing.png "window fitting"
[image5]: ./output_images/heat10.png "10 Frame heatmap"
[image7]: ./output_images/boxes.png
[video1]: ./outF.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project that was used as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  I used 8792 vehicle images and 8968 non-vehicle images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes, respectively:

![alt text][image1]
![alt text][image1a]

The code flow for this step is:
skimage.feature.hog() is called by get_hog_features() [file `lesson_functions.py` at line 17], which is called three times (ALL channels) by function extract_features() [file `lesson_functions.py` at line 102], which is called at lines 297 and 303 of `search_classify.py`.   

I then explored different color spaces and different `hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

A lot of combinations performed well on the training data, but then didn't work great on the test videos.

To get good performance on the videos, I used the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and ALL channels. 

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. The higher I went with the HOG orientations, the better, but limited it to 11. In the project walk through it was mentioned that there are diminishing returns above 9. Using ALL color channels was definitely working better than individual ones.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear support vector machine (SVM) using 9588 features, containing HOG, spatially binned color, and histograms of color. For color binning, the spatial size was 32x32 and histogram bins of 16. The features were scaled to zero mean and unit variance on lines 310 to 314 of `search_classify.py`.  

80 percent of the example images we randomly selected for training the classifier and the remaining 20 percent were used for testing the accuracy. The resulting test accuracy was 98.9% and it took 7.0 seconds to train. Other combinations had higher accuracy, but didn't perform as well on the videos.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was done in function find_cars() [starting at line 125 of `search_clasify.py`]. The HOG features were run once for each image to reduce computations. The y direction was limited to pixels from 400 to 656 to so that vehicles wouldn't be detected in the sky. The entire x range was searched. A square window of 64 pixels was used and the window was stepped 2 cells per slide, for an effective overlap of 75 percent.  

Using just one scale didn't detect enough cars. Using a scale much less than 1 had too many false positives. Two scales of 1.2 and 2 worked well, but 3 scales of 1, 1.5 and 2 worked better. 


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Originally, I was getting good training metrics, but not picking up enough of the cars in the video. Adding multiple scales helped greatly (see above). Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./outF.mp4)  There is a brief possible false positive at 42s in the shade, but oncoming cars are also picked up.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the heatmaps of detections in each frame of the video.  From the collection I created a cumulative heatmap over the last 10 frames and then thresholded that map (>2) to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This method smoothed out the boxes and reduced false positives, which tended to be noisy.

### Here is a cummulative heatmap over 10 frames at around the 0:29s mark of the video:

![alt text][image5]


### Here the resulting bounding boxes are drawn onto a frame at around 0:29s in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest two obstacles I faced on this problem were:
 
- getting the training features consistent with the implementation features. I had redundant conflicting parameters. 
- getting the VideoFileClip() function to run more than once per session. I added explicit handle close calls and reported it on the forum.

The approach used has a problem detecting a car that is obscured by another car and tends to bunch neighboring vehicles together. Also, a car with a vastly different velocity relative to the camera won't be detected as well by the averaging employed. I think that adjusting the scaling and averaging could help.

It would be good to combine this with the lane lines estimates and a relative vehicle speed estimate for each blob. Also, I started the suggested Vehicle class to organize/track vehicle detections, but didn't pursue it. This is a good approach to getting more intelligent information about the estimates.  Finally, I would be interested in trying a deep learning approach to see if it has any advantages.

 

