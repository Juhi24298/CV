#Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Initalziing the CNN
classifier=Sequential()

#Step-1 Convolution
classifier.add(Convolution2D(32,(3,3),activation='relu',input_shape=(128,128,3)))

#Step-2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Extra convolution and pooling
classifier.add(Convolution2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 Flatenning
classifier.add(Flatten())

#Step-4 Fully Connected layer
classifier.add(Dense(units=128,activation="relu"))

#Output layer
classifier.add(Dense(units=1,activation="sigmoid"))

#Compiling the fully connected layer
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Fitting section

#Importing ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

#Training data generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

#Training set
training_set = train_datagen.flow_from_directory(
        'C:/Users/Juhi/Downloads/cats_dogs/dogs-cats-images/dataset/train',#Enter your dataset path here
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

#Test set
test_set = test_datagen.flow_from_directory(
        'C:/Users/Juhi/Downloads/cats_dogs/dogs-cats-images/dataset/test',#Enter your dataset path here
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

#Fitting the classifier
classifier.fit_generator(
        training_set,
        steps_per_epoch=10,
        epochs=15,
        validation_data=test_set,
        validation_steps=10)


print("Training complete. Model is ready to classify")

#Check whether an image is a cat or dog
import numpy as np
from keras.preprocessing import image

samples=["C:/Users/Juhi/Downloads/cat.2.jpg",
         "C:/Users/Juhi/Downloads/dog101.jpg"]
#Accepting the path name as input

for i in range (0,2):
    path_name=samples[i]
    #Reading the image
    single_image=image.load_img(path_name,target_size=(128,128))
    #Converting the image to an array
    single_image=image.img_to_array(single_image)
    #Adding an extra dimension the image array
    single_image=np.expand_dims(single_image,0)
    #Prediciting the result
    category=classifier.predict(single_image)
    #Know which integer represent a cat or dog
    training_set.class_indices

    #Converting the int to string result
    if(category>0.5):
        print("Dog")
    else:
        print("Cat")
