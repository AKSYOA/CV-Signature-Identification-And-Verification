# from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import tensorflow as tf
from opt_einsum.backends import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential

def get_batch(triplets_list, batch_size, preprocess=False):
    batches_count = len(triplets_list) // batch_size
    triplet_index = 0
    bateches_counter = 0
    while bateches_counter < batches_count:  # no of batches
        j = bateches_counter * batch_size  # index of first triplet in the batch
        while j < (bateches_counter + 1) * batch_size and j < len(triplets_list):
            anchor = []
            positive = []
            negative = []
            j += 1
            anch, posit, negat = triplets_list[triplet_index]
            triplet_index += 1
            anch = np.array(anch, dtype=object)
            posit = np.array(posit, dtype=object)
            negat = np.array(negat, dtype=object)
            # if preprocess:
            #     anch = preprocess_input(anchor)
            #     posit = preprocess_input(positive)
            #     negat = preprocess_input(negative)
            anchor.append(anch)
            positive.append(posit)
            negative.append(negat)
        yield ([anchor, positive, negative])
        # batches[bateches_counter][increment]=triplets_list[triplet_index]
        bateches_counter += 1


def cnn_model(image_size):
    pretrained_model = Xception(
        input_shape=image_size,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )
    for i in len(pretrained_model) - 27:
        pretrained_model.layers[i].trainable = False
        model = Sequential([
            pretrained_model,
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(512, activation='relu'),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Dense(256, activation="relu"),
            tensorflow.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ], name="Encode_Model")

    return model

def get_siamesNetwork(image_size=(128,128)):
    model = cnn_model(image_size)               # make model
    anchor = tensorflow.keras.layers.Input(input_size=(image_size, image_size), name='anchor')
    positive = tensorflow.keras.layers.Input(input_size=(image_size, image_size), name='positive')
    negative = tensorflow.keras.layers.Input(input_size=(image_size, image_size), name='negative')
    output_anchor=model(anchor)                 #inputLayers to madel
    output_positive=model(positive)
    output_negative=model(negative)
    output = np.concatenate([output_anchor,output_positive,output_negative],axis=1)
    siamese_network = Model(
        inputs  = [anchor, positive, negative],
        outputs = output,
        name = "Siamese_Network"
    )


