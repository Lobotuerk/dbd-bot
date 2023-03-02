import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import cv2
from random import shuffle
import numpy as np

local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

FILE_I_END = 65

MODEL_NAME = 'INCV3_WEIGHTS.h5'

LR = 1e-3
EPOCHS = 50
WIDTH = 480
HEIGHT = 270

pre_trained_model = InceptionV3(input_shape = (3, WIDTH,HEIGHT),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(256, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(6, activation='sigmoid')(x)
model = Model( pre_trained_model.input, x)

model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

total_acc = np.array([])
total_val_acc = np.array([])
total_loss = np.array([])
total_val_loss = np.array([])

for e in range(EPOCHS):
    print('EPOCH NUMBER ', e)
    epoch_acc = np.array([])
    epoch_val_acc = np.array([])
    epoch_loss = np.array([])
    epoch_val_loss = np.array([])
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):

        try:
            file_name = './datanew/training_data-{}.npy'.format(i)
            train_data = np.load(file_name, allow_pickle=True)
            print('training_data-{}.npy'.format(i), e, '{}/{}'.format(count,FILE_I_END))
            shuffle(train_data)
            train = train_data[:-50]
            test = train_data[-50:]
            X = np.array([i[0] for i in train])
            X = X.reshape(X.shape[0],3,WIDTH,HEIGHT)
            Y = np.array([i[1] for i in train])

            test_x = np.array([i[0] for i in test])
            test_x = test_x.reshape(test_x.shape[0],3,WIDTH,HEIGHT)
            test_y = np.array([i[1] for i in test])
            # print(model.summary())
            # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            history = model.fit(x=X, y=Y, epochs=1, validation_data=(test_x,test_y), batch_size=25)
            acc = history.history['acc']
            epoch_acc = np.append(epoch_acc, acc)
            val_acc = history.history['val_acc']
            epoch_val_acc = np.append(epoch_val_acc, val_acc)
            loss = history.history['loss']
            epoch_loss = np.append(epoch_loss, loss)
            val_loss = history.history['val_loss']
            epoch_val_loss = np.append(epoch_val_loss, val_loss)

            if count%10 == 0:
                print('SAVING MODEL!')
                model.save_weights(MODEL_NAME)

        except Exception as ex:
            print(str(ex))
    total_acc = np.append(total_acc, np.average(epoch_acc))
    total_val_acc = np.append(total_val_acc, np.average(epoch_val_acc))
    print("total :",total_val_acc)
    print("actual :",np.average(epoch_val_acc))
    total_loss = np.append(total_loss, np.average(epoch_loss))
    total_val_loss = np.append(total_val_loss, np.average(epoch_val_loss))

total_acc = np.array(total_acc)
# print(total_acc)
total_val_acc = np.array(total_val_acc)
# print(total_val_acc)
total_loss = np.array(total_loss)
# print(total_loss)
total_val_loss = np.array(total_val_loss)
# print(total_val_loss)

np.save('./total_acc.npy', total_acc)
np.save('./total_val_acc.npy', total_val_acc)
np.save('./total_loss.npy', total_loss)
np.save('./total_val_loss.npy', total_val_loss)
