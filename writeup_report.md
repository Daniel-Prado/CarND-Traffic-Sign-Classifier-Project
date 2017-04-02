# **Traffic Sign Recognition** 



#### by Daniel Prado Rodriguez
#### Udacity SDC Nanodegree. Feb'17 Cohort.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "first image per class"
[image2]: ./examples/image2.png "first 50 images per random class"
[image3]: ./examples/image3.png "class histograms of training, validation and test sets"
[image4]: ./examples/image4.png "class histogram of training, after augmentation"
[image5]: ./examples/image5.png "augmentation example"
[image6]: ./examples/image6.png "normalization example"
[image7]: ./examples/image7.png "softmax probabilities"
[image8]: ./examples/image8.png "Layer 1 output feature maps"
[imageSignal1]: ./new_images/image_class04.jpg "Roundabout mandatory"
[imageSignal2]: ./new_images/image_class18.jpg "Children crossing"
[imageSignal3]: ./new_images/image_class28.jpg "Ahead only"
[imageSignal4]: ./new_images/image_class35.jpg "General caution"
[imageSignal5]: ./new_images/image_class40.jpg "Speed limit (70km/h)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Daniel-Prado/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Here you can also check the code output of the python notebook, as an HTML file. [code output](https://github.com/Daniel-Prado/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html)

### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the Step 1 section of my IPython notebook.  
I used standard python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
The size of the training set is 34799 samples
* The size of test set is ?
The size of the test set is 12630 samples.
Note that a Validation set is also provided with 4410 samples.
* The shape of a traffic sign image is ?
(32, 32, 3) , that is 32x32 pixels x 3 channels per pixel.
* The number of unique classes/labels in the data set is ?
The number of classes is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cells of Section 1 of the IPython notebook.
I have done the following exercises:
* Show the 1st sample found in the Training set for each class.

![alt text][image1]

* Show the 1st 50 samples in the Training Set of a randomly choosen class.

![alt text][image2]

* Show the histogram distribution per classes of the Training set, Validation set and Test set.

![alt text][image3]

* Discuss the comparison of the Training vs. Validation histograms and some remarks about the given valid.p set.



### Design and Test a Model Architecture

##### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Note: Before applying normalization to the training set, I applied data augmentation, explained later in this report.

**DATA NORMALIZATION**
I have experimented several methods for data normalization, but finally I have chosen the following steps:
1) Convert all the images from RGB to YUV colorspace (Finally, I have used only the Y luminance component - grayscale)
2) Apply the CV2.equalizeHist function to equalize the histogram of every image.
3) Normalize the image by substracting the Mean of each image, and dividing by the Standard Deviation of each image.

The 3rd step guarantees that every resulting image has a standar deviation of 1.0 and a Mean of 0.0... However this does not guarantee that a given pixel of the image is constrained within given margins (unlike for example, substracting 128 and dividing by 255)...  Even like that, I have observed empirically that this approach produces better validation results.

Note that all the steps described are applied to all the training, validation and test sets.

As an example, here is a sample of a pre-processed and normalized image:

![alt text][image6]


##### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

First of all, it must be understood that unlike previous SDC ND cohorts, I did not need to split the data into training and validation sets, because the validation set is already provided in a separate valid.p file.

As discussed in the notebook in Section 1, I experimented joining the training and validation sets, shuffling the samples and obtaining a new validation set. Doing that, the performance (accuracy) of the classifier augmented greatly for the validation set. 

**However, taking into account that we were required to get a minimum validation accuracy of 0.93, this obviously applies to the valid.p file that we are given, not any other set that we could construct. So, I have stuck to the valid.p given.**

**DATA AUGMENTATION**
The first thing I do with the dataset is to perform data augmentation. I will augment the data by applying random Translation, Rotation and Shear to the training images. I will augment so that every class has at least 1000 samples. This means that some of the classes will not require augmentation, whilst others will have an important increase of samples. I will not augment the data in terms of brightness or contrast, I think the samples provide enough variability on that sense.

Note that data augmentation applies only to the training set.

See my code for data augmentation in the cell code under section "Data Augmentation"
The resulting new histogram of the training set is shown below:

![alt text][image4]

Also, a sample of how the augmentation works for a given image, the following exercise is shown:

![alt text][image5]

My final training set had 57028 number of images. My validation set and test set had 4410 and 12630 number of images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located under the Header "Model Architecture" of section 2 of the ipython notebook. 

My Architecture (MyLeNet) is based basically on the LeNet-5 used in the class lab, introducing the following modifications:
* Increasing L1 output size from 6 to 12.
* Increasing L2 output size from 16 to 32.
* Modified following layers accordingly to the new L1, L2 sizes.
* Introducing Dropout in training phase, between the FC layers, with a 0.5 probability, that worked better than 0.25 or 0.75.

I also experimented with other enhancements, like adding a new convolutional or fully connected layer, increasing FC layer feature sizes, and even introducing the concept of multi-connected conv layers (concatenating the flattened output of conv1 to the FC0 layer. However, none of these changes improved the results, so I removed them and kept the simple design.

Hence, finally my model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y-luminance (grayscale) image   	    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					| Relu activation function.						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					| Relu activation function.						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 (800 flatened)	|
| Fully connected 1		| Input 800, output 120  						|
| RELU					| Relu activation function.						|
| DROP-OUT              | Drop-out with probability 0.5                 |
| Fully connected 2		| Input 120, output 84							|
| RELU					| Relu activation function.						|
| DROP-OUT              | Drop-out with probability 0.5                 |
| Fully connected 3		| Input 84, output 43 (LOGITS)					|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters: A value between 25 and 30 Epochs seems to work well with my model, reaching almost a 99.9% training accuracy with almost 98% validation accuracy and 95-96% Test accuracy. For every Epoch, the training and validationa accuracies are calculated.
Decreasing slightly the default learning rate from 0.001 to 0.00095 seemed to improve the results.
I have tried to apply an exponentially decreasing learning rate, but this seems not to have much effect with the AdamOptimizer, that converges much faster than simple Gradient Descent, and seems to manage the learning rate adapting internally.

A batch size of 128 seemed to perform better than other values (256)

The code for training the model is located under the "Training Pipeline" and "Train the Model" headers.


##### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located under the Model Evaluation header in the Ipython notebook.
I also present further analysis after the training is complete under header "Evaluate Results in Validation per every class".

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.981
* test set accuracy of 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

As explained before, my model is based on the LeNet-5 shown in the class lab, with some enhancements.

* What were some problems with the initial architecture?

The initial architecture seemed to lack enough complexity to analyze the features of all the traffic signs classes, hence showing Underfitting. This could be normal as the LeNet-5 was designed to classify digits (10 classes) instead of 43 classes.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Adding more features to the first convolutional layers was the key step to make the model more complex to adapt to the traffic signals complexity.
Adding the Dropout layers clearly improved the accuracy, by making the model more "sure" of the predictions.

* Which parameters were tuned? How were they adjusted and why?

Already explained above. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Already explained above.

If a well known architecture was chosen:
* What architecture was chosen?

LeNet-5 

* Why did you believe it would be relevant to the traffic sign application?

Well, in the class it was shown that LeNet could achieve a good performance right away, so with some enhancements I was sure I could achieve an excellent accuracy.
Of course there are more sofisticated and state-of-the-art architectures out there that I would love to try, but I wanted to focus to understand how a basic "classic" convnet architectute works first.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The training accuracy reaches almost 100%, which maybe could show a bit of overfitting.
Anyway, the validation accuracy is excellent (98.1%) and the test accuracy does not drop too much (95.5%). I think those are good results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have obtained these images myself from Google StreetView in cities such as Mannheim and Cologne in Germany... Actually any Western European country would have server, as these signals are 
Here are five German traffic signs that I found on the web:

![alt text][imageSignal1] ![alt text][imageSignal2] ![alt text][imageSignal3] 
![alt text][imageSignal4] ![alt text][imageSignal5]

The first image would probably be the least difficult to classify because it is centered, with a simple background, well illuminated and with all its features clearly shown.
The second image is also clearly identifiable for persons, however it is off-centered, and it has a strong background pattern of lines that could interfere with the signal line features.
The third image shape is highly distorted.
The fourth image is off-centered, slightly distored and rotated.
The fifth image is off-centered and has lost colour contrast.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the section "Predict the sign typ of each image" of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Roundabout mandatory     		| Roundabout mandatory   									| 
| Children crossing      		| Children crossing  										|
| Ahead only					| Ahead only										|
| General caution      		    | General caution					 				|
| Speed limit (20km/h)		    | Speed limit (70km/h)|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This result is broadly compatible with an accuracy of the test set of 95% percent. Of course the fact that the number of these images is only 5, means that only 1 mistake would decrease accuracy from 100% to 80%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the section "Output Top 5 Softmax Probabilities" of the Ipython notebook.

In the notebook screenshot below we can observe the softmax probabilities represented in a bar chart for each image.
For the first 4 images we see that the classifier is absolutely confident in the predictions with a 100% probability.
Only for the 5th image, we see that the classifier is very uncertain. Only giving a 50% probability to the most likely prediction of 20km/h speed limit. whilst the correct answer is assigned only about an 8%.

![alt text][image7]

#### 4. Visualize the Neural Network's State with Test Images
In this optional section I have managed to represent visually the feature maps of the 1st convolutional layers when the model receives the input of the 5 test images used in the previous sections.
It is very interesting how the first CONV layer output extracts the feature maps of the 5 test signals. And more interestingly even for me, is how each feature number across the 5 test signals seems to focus in some specific common feature. For example Feature 0 seems to focus on "left borders" while Feature 8 seems to focus on "top-right" borders.
This also demonstrates that the MyLeNet network learned to detect useful traffic sign features on its own. We never explicitly trained it to detect the signal's round or triangular shapes, numbers, figures, etc.

The maps areshown in the following figure:

![alt text][image8]



