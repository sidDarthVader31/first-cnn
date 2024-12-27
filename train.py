# import the libraries 
import tensorflow as tf
from  tensorflow.keras.preprocessing.image import
ImageDataGenerator

print(tf.__version__)

#step 1 - data pre processing 


#processing the training set 

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary'
        )

#preprocessing the test set
#we are not adding additional transformation here because we
do not want to loose information of our test data set 

test_datagen = ImageDataGenerator(rescale = 1/.255)
test_set = test_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32, 
        class_mode = 'binary'
        )

# Building the CNN 

#initializing the CNN 
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = [64,64,3])
    ])

#convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                               activation='relu'))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


#add a second convolution layer 
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                               activation='relu'))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#fully connected layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#output layer 
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# training the CNN

#compile the convolutional neural network
cnn.compile(optimizer='adam', loss= 'binary_cross_entropy',
            metrics=['accuracy'])

#train the CNN
cn.fit(x= training_set, validation_data= test_set, epochs=35)


#Making a single prediction 

import numpy as np 

from tensorflow.keras.preprocessing import image 

test_image = image.load_img('/path/to/image', target_size
                            =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)

print(result)
if result[0][0]:
    prediction = 'dog'
else:
        prediction = 'cat'

print(prediction)
