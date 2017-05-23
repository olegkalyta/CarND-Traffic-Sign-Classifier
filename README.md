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

If an iterative approach was chosen:
##### Which parameters were tuned? How were they adjusted and why?
I experimented with `BATCH_SIZE` and saw, `32` works best for this model
I experimented with `learning_rate`. When I decreased it, accuracy was worse, so I increased it a bit to `0.00115` and it shown good results.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]

The first image might be difficult to classify because ...

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

[ 5  3  1 38 11]

TopKV2(values=array(

    [[ 19.29938889,   1.34492826,  -0.45026189,  -4.35485888,
         -4.64507008],
       [ 18.41223335,  16.99563217,   8.6574831 ,   7.67988777,   3.6240077 ],
       [ 15.68963051,   6.95158434,   5.9887023 ,  -1.9844352 ,
         -2.86871958],
       [ 31.66697884,  18.21386147,  12.99901676,  11.73639965,   8.1025362 ],
       [ 30.51384544,  26.96464157,  23.40794563,   3.66663933,
         -5.31201315]], dtype=float32), indices=array([[ 5,  7, 31, 40, 20],
       [ 3,  1,  6,  5,  2],
       [ 1, 25, 18,  0, 31],
       [38, 33, 36, 40, 39],
       [11, 18, 26, 27, 30]])
   )