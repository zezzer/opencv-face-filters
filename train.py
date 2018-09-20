from model import *
from preprocess import *
import numpy as np

num_epochs = 10
batch_size = 50
seed = 1

np.random.seed(seed)

print("Reading in data")
train_data, train_labels = get_training_data() 

to_randomize = np.hstack((train_data, train_labels))
np.random.shuffle(to_randomize)
train_data = to_randomize[:, :-30]
train_labels = to_randomize[:, -30:]

train_data = train_data.astype('float32')/255
train_data = train_data.reshape(train_data.shape[0], 96, 96, 1)
train_labels = (train_labels.astype('float32')-48)/48

print("Creating model")
model = create_model()

print("Training")
train_model(model, train_data, train_labels, num_epochs, batch_size)

save_model(model, 'facial_keypoint_model.h5')
