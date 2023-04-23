import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = y_train // 5 # Map to 20 superclasses
y_test = y_test // 5

# Define the neural network architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax')) # 20 superclasses

# Compile the model with appropriate loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks to save the best model during training
best_model_file = "best_model.hdf5"
checkpoint = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
batch_size = 128
epochs = 100
history = model.fit(x_train, keras.utils.to_categorical(y_train, 20), batch_size=batch_size, epochs=epochs,
                    validation_data=(x_test, keras.utils.to_categorical(y_test, 20)), shuffle=True,
                    callbacks=[checkpoint])

# Evaluate the model on the test set
model.load_weights(best_model_file) # Load the weights of the best model
scores = model.evaluate(x_test, keras.utils.to_categorical(y_test, 20), verbose=1)
print("Test accuracy:", scores[1])




