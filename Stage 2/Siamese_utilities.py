# from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import tensorflow as tf
from random import shuffle

from tensorflow.keras import backend, layers, metrics
from keras.applications.xception import Xception
from opt_einsum.backends import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential


def get_triplet(data):
    triplets = []
    classes = ['A', 'B', 'C', 'D', 'E']
    for i in range(len(classes)):
        Class = classes[i]
        for image in data:
            if image[2].__contains__(Class):
                anchor = image
                label_anchor = image[1]
                name = image[2]

        for pos in data:
            if pos[2].__contains__(Class) and label_anchor == pos[1] and name != pos[2]:
                positive = pos
                for neg in data:
                    if neg[2].__contains__(Class) and label_anchor != neg[1]:
                        negative = neg
                        # print("anchor", label_anchor, " ", name)
                        # print("postive", pos[1], " ", pos[2])
                        # print("negative", neg[1], " ", neg[2], "\n")
                        triplets.append([anchor, positive, negative])
                        shuffle(triplets)
    return triplets


def get_batch(triplets_list, batch_size, preprocess=False):
    batch_steps = len(triplets_list) // batch_size

    for i in range(batch_steps + 1):
        anchors = []
        positives = []
        negatives = []

        j = i * batch_size  # index of first triplet in the batch
        while j < (i + 1) * batch_size and j < len(triplets_list):
            anchor, positive, negative = triplets_list[j]

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
            j += 1

        # if preprocess:
        #     anch = preprocess_input(anchors)
        #     posit = preprocess_input(positives)
        #     negat = preprocess_input(negatives)
        anchors = np.array(anchors, dtype=object)
        positives = np.array(positives, dtype=object)
        negatives = np.array(negatives, dtype=object)

        yield [anchors, positives, negatives]


def get_encoder(image_size):
    pretrained_model = Xception(
        input_shape=image_size,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )
    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

        encode_model = Sequential([
            pretrained_model,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ], name="Encode_Model")

    return encode_model


class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance


def get_siamese_network(image_size=(128, 128, 3)):
    encoder = get_encoder(image_size)

    # Input layers
    anchor_input = layers.Input(image_size, name='Anchor_Input')
    positive_input = layers.Input(image_size, name='Positive_Input')
    negative_input = layers.Input(image_size, name='Negative_Input')

    # Encoded Vectors
    encoded_anchor = encoder(anchor_input)
    encoded_positive = encoder(positive_input)
    encoded_negative = encoder(negative_input)

    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network
