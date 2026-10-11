from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
#from google.colab.patches import cv2_imshow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


num_train = 28707
num_val = 7174
batch = 32
num_of_epoch = 20

# Pre-Porcessing of and converting into gray-scale images
data_train = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

valid_gen = val_datagen.flow_from_directory(
        'data/test',
        batch_size=batch,
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='categorical')

training_gen = data_train.flow_from_directory(
        'data/train',
        batch_size=batch,
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='categorical')

# Model Initialization
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# Computing loss and saving weights
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model_info = model.fit_generator(
training_gen,
validation_data=valid_gen,
epochs=num_of_epoch,
validation_steps =num_val // batch)
model.save_weights('model.h5')
