# **Traffic Sign Recognition**

## Writeup Template

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.png "Visualization of training dataset"
[image2]: ./test_images/1.jpg "Traffic Sign 1"
[image3]: ./test_images/2.jpg "Traffic Siпn 2"
[image4]: ./test_images/3.jpg "Traffic Sign 3"
[image5]: ./test_images/4.jpg "Traffic Sign 4"
[image6]: ./test_images/5.jpg "Traffic Sign 5"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **12630**
* The size of test set is **43**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **60059** (overall test, validataion & training sets).

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

##### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will decrease input size on first layer from 32x32x3 to 32x32x1. But I using this method I lost prediction accuracy.
So I decided to follow another offered option - use approximately normalization of images: `(pixel - 128)/ 128`.
I created [normalizeInput.py](../master/normalizaInput.py) to do it once overall all input data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			     	|
| Convolution 2  	    | 2x2 stride, valid padding, output = 10x10x16  |
| Max pooling		    | Input = 10x10x16. Output = 5x5x16.        	|
| Fully Connected 1		| Input = 400. Output = 120      				|
| Fully Connected 2		| Input = 120. Output = 84      				|
| Fully Connected 3		| Input = 84. Output = 43												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an proposed LaNet architecture and did not change anything.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.954
* test set accuracy of ?

###### What was the first architecture that was tried and why was it chosen?
I used LaNet network from prev lesson
###### What were some problems with the initial architecture?
I did not found any problems with this architecture
###### How was the architecture adjusted and why was it adjusted?
No changes except labels number 43 instead of 10
######  Which parameters were tuned? How were they adjusted and why?
I experimented with `BATCH_SIZE` and saw, `32` works best for this model
I experimented with `learning_rate`. When I decreased it, accuracy was worse, so I increased it a bit to `0.00115` and it shown good results.

###### What are some of the important design choices and why were they chosen?
etc. ?
Regarding to model architecture, I did not make any design changes

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]

The images might be difficult to classify because they another  brightness and contrast, plus sign itself where bigger then training and validation data.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No stopping      		| Speed limit (80km/h) 									|
| **Speed limit (60km/h)** 	| **Speed limit (60km/h)**										|
| Pedestrian crossing	| Yield											|
| Stop and give way	   	| Keep right				 				|
| General caution		| Right-of-way at the next intersection      							|


Model accuracy on new images is **20%**

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

    [[1.000 0.000 0.000 0.000 0.000]
     [0.805 0.195 0.000 0.000 0.000]
     [1.000 0.000 0.000 0.000 0.000]
     [1.000 0.000 0.000 0.000 0.000]
     [0.971 0.028 0.001 0.000 0.000]]

   My model predicted correctly only second sign. I think the reason is in it's simplicity, it's very schematic and has nice contrast, better then in first sign.

   Images 1, 3, 4 are predicted with very low accuracy, there are basically no other competitors in softmax predictions for these images.

   I think this is wrong. And it definitely may be improved in future.

#### **More details on results:**

   Sign1 - No stopping
   Prediction -
   **Speed limit (80km/h)**
   Speed limit (100km/h)
   Wild animals crossing
   Roundabout mandatory
   Dangerous curve to the right

   Sign 2 - Speed limit (60km/h)
   Prediction -
   **Speed limit (60km/h)**
   Speed limit (30km/h)
   End of speed limit (80km/h)
   Speed limit (80km/h)
   Speed limit (20km/h)

   Sign 3 - Pedestrian crossing
   Prediction -
   **Speed limit (30km/h)**
   Road work
   General caution
   Speed limit (20km/h)
   Wild animals crossing

   Sign 4 - Stop and give way
   Prediction -
   **Keep right**
   Turn right ahead
   Go straight or right
   Roundabout mandatory
   Keep left

   Sign 5 - General caution
   Prediction -
   **Right-of-way at the next intersection**
   General caution
   Traffic signals
   Pedestrians
   Beware of ice/snow