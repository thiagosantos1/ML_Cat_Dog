# Conception

	* What are Convolutional Neural Networks?
		* The main idea it's very similar to ANN.
		* How The CNN can identify caracteristics in images?
			* each pixel is a "box"/filter of a choseen dimension, usually 3x3.
			* and for each pixel, the color of that pixil goes from 0 to 255, for a color representation(black or white).
			* For colorful pictures, there are 3 boxes(3 dimension), and for each, the pixel also goes from 0 to 255.

	* Steps for the CNN 
		1) Convolution (X) Layer 
			* Main idea --> Find features in a image and make the image smaller by removing not necessary pixels 
			* It is baasically a combined integration of the two functions and it shows you how one function modifies the other.   
			* A convolution operation is a circule with an X inside. (X)  
			* But what exactly is it?
				* From the image, we process/filter the image with a feature/filter detector, using a filter.
				* From the original image, we use the filter to get to a filter/feature Map. How?
					* We do a Matrix multiplixation. Thus, from the size of the filter, we get a part of the original 
					* image, with same size, and do a matrix multiplication. 
					* Then, we sum up this multiplication, and the result goes to the feature map matrix.
					* We keep this loop for all squares inside of the original image, that has same size of the filter

			* We do lose some information from the image by filtering. 
			* However, the process also tries to find patterns, and therefore, we do not lose the main pixels, used to classify the picture. Because highest number represents a pattern from a feature, and we keep the highest number

			* We do not use only one filter map, but many. Therefore, we get multiple results of feature maps. 
			* That way we avoid losing important informations. 
			* Thus, each feature map can have some important information from the picture.
				* Example: 
					* One may have a year
					* Or a mouth
					* Or a Nouse, etc.

		2) ReLU Layer --> Rectifier - Same layer as 1
			* After having all the new feature maps, we want to apply the ReLU.
			* It is important because we want to increase non-linearity in ou CNN.
				* Why?
					* Because images are not linear. There are different colors, borders, and etc.
					* However, by runnig a filter and creating a feature map, we may be creating something
					* Linear, and therefore we need to break up the linearity.

		3) Max Pooling Layer - Downsampling 
			* Benefits
				* Spacial Invarience
				* Reducing the size by 75%
				* Preserving features
				* Reducing number of paramters that would go to our final layer
				* Avoid overfitting.

			* What is it?
				* An image can variety its position and shape. How can the CNN identify an image, undepending the light, positions, shadow, etc? We use Pooling
			*  After the Convolution layer, we run Max Pooling.
			* We use the same idea of filter, but this time we get a box(size of your choice - usually 2x2)
			* Then, we get the highest value in the box and set in a cell of the pooled feature Map.
			* Thus, if feature map is a matrix 5x5 and you choose a box of 2x2, pooled feature map would be 3x3
			* As same as Filtering, here we do keep the highest number. Thus, we are keeping what seems like to be
			* a feature, rather than a noise or a different position, etc. 
			* Thus, we are getting rid of 75% of the information that is not the feature/pattern, that we are looking at.
			* Also, if a different position, or light or parts of the body is different from image to image.
			* By applying pooling, we avoid to miss that, because if it is a important feature, it is gonna have a hight number in your pixel.
			* And as said before, we do keep the highest. Thus, it does't matter if a same feature from different images.
			* Are saved in a different position of the matrix, at the end, it would be in the pooled feature map matrix as well.

		4) Flattening
			* Here, the only thing done is to get the matrix, and turn it to a vector/flat. 
			* Why?
				* Because CNN is basically just to process the image
				* The output of CNN(Matrix) is turned to a vector, which is used as input layer for a ANN.

		5) Full Connection
			* Main idea
				* We add a whole artificial neural network to our convolutional neural network.

			* Then the idea is similar to ANN. We run the foward propagation, the net makes a guess, and we calculate the loss function.
			* Then, it backpropagates to ajust the weights as normal.
			* However, here, we also ajust the feature values, because we may be looking for the wrong one.







