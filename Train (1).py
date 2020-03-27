import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
#import seaborn as sns
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from  keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
#from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

datagen = ImageDataGenerator(
        preprocessing_function = keras.applications.mobilenet.preprocess_input,
        featurewise_center = False,
        samplewise_center=False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range=45,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = False,
        shear_range = 0.2)

train_path = "data2"
valid_path = "data"

train_batches = datagen.flow_from_directory(train_path, target_size=(224,224), classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'], batch_size = 60)
valid_batches = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224, 224), classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'], batch_size = 60)

def plots(ims, figsize=(12,6), rows = 1, interp = False, titles = None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize = 16)
        plt.imshow(ims[i], interpolation = None if interp else'none')

imgs, labels = next(train_batches)
mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()

x = mobile.layers[-6].output
predictions = Dense(15, activation='softmax')(x)
model = Model(inputs = mobile.input, outputs = predictions)

for layer in model.layers[: -23]:
    layer.trainable = False

model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(train_batches, steps_per_epoch = 18, validation_data = valid_batches, validation_steps = 2, epochs = 15, verbose = 2)
model.save("model.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
