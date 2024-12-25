#############################
#Coding the image classifier#
#############################

###########
#Libraries#
###########
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

#################################################################################################################################################################

#Defines relative directory
data_dir = 'Path'

#Defines data directories for exploitable and defective images
exploitable_dir = os.path.join(data_dir, 'exploitable')
defective_dir = os.path.join(data_dir, 'defective')

#Defines output directories
training_dir = os.path.join(data_dir, 'training')
testing_dir = os.path.join(data_dir, 'testing')

#Creates training and testing directories
os.makedirs(os.path.join(training_dir, 'exploitable'), exist_ok=True)
os.makedirs(os.path.join(training_dir, 'defective'), exist_ok=True)
os.makedirs(os.path.join(testing_dir, 'exploitable'), exist_ok=True)
os.makedirs(os.path.join(testing_dir, 'defective'), exist_ok=True)

#Defines percentage of data to use for training
train_percentage = 0.8

#Lists images in exploitable and defective classes
exploitable_images = os.listdir(exploitable_dir)
defective_images = os.listdir(defective_dir)

#Randomly shuffles the image lists
random.shuffle(exploitable_images)
random.shuffle(defective_images)

#Calculates the number of images to use for training and testing
exploitable_train_count = int(train_percentage * len(exploitable_images))
defective_train_count = int(train_percentage * len(defective_images))

#Copies images to the training and testing directories
for i in range(exploitable_train_count):
    source_file = os.path.normpath(os.path.join(exploitable_dir, exploitable_images[i]))
    shutil.copy(source_file, os.path.normpath(os.path.join(training_dir, 'exploitable', exploitable_images[i])))

for i in range(defective_train_count):
    source_file = os.path.normpath(os.path.join(defective_dir, defective_images[i]))
    shutil.copy(source_file, os.path.normpath(os.path.join(training_dir, 'defective', defective_images[i])))

for i in range(exploitable_train_count, len(exploitable_images)):
    source_file = os.path.normpath(os.path.join(exploitable_dir, exploitable_images[i]))
    shutil.copy(source_file, os.path.normpath(os.path.join(testing_dir, 'exploitable', exploitable_images[i])))

for i in range(defective_train_count, len(defective_images)):
    source_file = os.path.normpath(os.path.join(defective_dir, defective_images[i]))
    shutil.copy(source_file, os.path.normpath(os.path.join(testing_dir, 'defective', defective_images[i])))

#################################################################################################################################################################

#Creates data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(training_dir, target_size=(246, 226), batch_size=10, class_mode='binary', shuffle=True)
test_generator = test_datagen.flow_from_directory(testing_dir, target_size=(246, 226), batch_size=10, class_mode='binary', shuffle=False)

#################################################################################################################################################################

#Defines learning rate schedule function
def lr_scheduler(epoch):
    initial_learning_rate = 0.001
    decay = 0.5
    lr = initial_learning_rate * (decay ** epoch)
    return lr

#Creates an optimizer with the initial learning rate
optimizer = Adam(learning_rate=0.001)

#Sets up the learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_scheduler)

#Sets up the initial CNN model features
model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(246, 224, 3)), MaxPooling2D(2, 2), Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(2, 2), Conv2D(128, (3, 3), activation='relu'), MaxPooling2D(2, 2), Flatten(), Dense(128, activation='relu'), Dense(1, activation='sigmoid')])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=67, epochs=20, validation_data=test_generator, validation_steps=16, callbacks=[lr_scheduler])

#################################################################################################################################################################

#Saves the image classifier for future use
model.save('CNN_TrainedImageClassifier.h5')

#################################################################################################################################################################

#Plots training and validation accuracy values
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#plots training and validation loss values
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title("Model loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

############################
#Using the image classifier#
############################

###########
#Libraries#
###########
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array    

#################################################################################################################################################################
#Defines paths for everything: Trained classfier, data to be sorted, folders where the sorted data will be copied
model = tf.keras.models.load_model('Path to where the .h5 is stored')
input_folder = 'Path to unsorted images folder'
output_folder_exploitable = 'Path to where the classifier will store exploitable images'
output_folder_defective = 'Path to where the classifier will store defective images'
os.makedirs(output_folder_exploitable, exist_ok=True)
os.makedirs(output_folder_defective, exist_ok=True)

#################################################################################################################################################################

def classify_and_move_images(input_folder, output_folder_exploitable, output_folder_defective):    
    #Uses the model to classify images and move exploitable ones to the "figures" folder
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        if prediction[0][0]>0.5:
            shutil.copy(img_path, os.path.join(output_folder_exploitable, filename))
        else:
            shutil.copy(img_path, os.path.join(output_folder_defective, filename))

#Classifies then moves the images
classify_and_move_images('input folder', 'output_folder_exploitable', 'output_folder_defective')            

