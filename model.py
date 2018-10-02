from keras.models import load_model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

def create_model(optimizer='adam', loss='mean_squared_error'):
	model = Sequential()
	model.add(Convolution2D(32, (3, 3), input_shape = (96, 96, 1), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Dropout(0.25))

	model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Dropout(0.5))
	model.add(Dense(30))

	model.compile(optimizer=optimizer, loss=loss)

	return model;

def train_model(model, training_data, training_labels, epochs, batch_size):
	return model.fit(training_data, training_labels, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, shuffle=True)

def save_model(model, file_name):
	model.save(file_name)

def load_cnn(file_name):
	return load_model(file_name)