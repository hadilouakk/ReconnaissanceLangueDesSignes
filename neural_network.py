import tensorflow.keras as keras 
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
train_dir =  'data'

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir)
print(next(train_generator)[0].shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd")

model.fit(train_generator, epochs=20)
model.save('model1.h5')