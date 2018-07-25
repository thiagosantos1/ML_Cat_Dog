# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Paramters
    # In the add(layer) we choose the step to do, like convolution, pooling or dense(for ANN)
    # Convolution2D(nb_filer, nb_rows, nb_columns)
        # nb_filter = number of filters, number of features detectors = number of feature map to be created
            # We can play with this number
        # nb_rows & nb_columns in the feature detector
        # How to choose the number ?
            # the practice is to add a convolutional layer with 32 3X3
            # Then, after pooling, add one more, with 64 3X3, to get more details of the picture
        # input_shape= () Input shape of your image. Size you want to work with
            # input_shape(width, hight, 3(dimension))
                # 3 dimension for colorful or 1 for black & white
        # activation --> Activation for that layer
            # porpouse --> Avoiding having negative values for pixels and linearity

    # MaxPooling2D --> To reduce the size of your feature map
        # pool_size(rowns, columns)
        # It is used to reduce the complexity, and the number of nodes for the dense layer(ANN)
        # We keep the "most" important features found it

    # Flatten()
        # To convert all our pool and create a vector. This will be the input of the ANN
        # How we dont lose the most important informations?
            # because we have run the filters and pooling, and we have kept the highest number
            # Which represents the most important features found so far.

    # Dense() --> Full conection layer
        # Dense(output_dim = number of nodes in the hidden layer, activation)
            # have to experiment with this numbers
            # the first one is the hidden layer, that gets the flatten output as input

            # the last layer is the output. sigmoid for binary or softmax for more categories

    # compile(optimizer = which gradiente descend, loss = how to calculate or error function, metrics)
        # adam = a stochastic gradient descend algorithm
        # loss = binary_crossentropy --> A better error function, for binary output
            # or categorical_crossentropy for more than one output

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# We have to pre-process the image, to avoid overfitting
# This function creates differents baches, with random images
# and Then it process them to avoid overfittin.
# How?
    # It can turn the pictures
    # rescale
    # and others
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Paramters --> We have to play with this numbers - called art
    # target_size --> Size of the img. Change here for better results(need GPU for bigger imgs)
    # batch_size --> divide the pictures for baches
    # class_mode --> Binary for 2 output or categorical
    # steps_per_epoch = total imgs traing/bach size
    # Epoch --> How many times we gonna repeat all the process for all images
    # validation_steps = total imgs test/bach size

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # same as before the in layers
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32))

# to test against a especific img
from skimage.io import imread
from skimage.transform import resize
import numpy as np

img = imread('/Users/thiago/Downloads/dog1.jpg') #make sure that path_to_file contains the path to the image you want to predict on.
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)

if(np.max(img)>1):
    img = img/255.0

prediction = classifier.predict_classes(img)

if(prediction):
    print ("DOG")
else:
    print ("CAT")



