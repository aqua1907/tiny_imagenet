import h5py
import numpy as np
import tensorflow as tf
from preprocess.utils import mean_preprocess
import json
from config import tiny_image_net_config as config


# this class is borrowed from the book "Deep Learning for computer vision with python. Practional Bundle", Adrian Rosebrock
class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, mean=True, aug=None, binarize=True, classes=200):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.mean = mean

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0
        with open(config.DATASET_MEAN) as json_file:
            means = json.load(json_file)

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                if self.binarize:
                    labels = tf.keras.utils.to_categorical(labels, self.classes)

                # if the data augmenator exists, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                if self.mean:
                    procImages = []
                    for image in images:
                        image = mean_preprocess(image, means["R"], means["G"], means["B"])

                        procImages.append(image)

                    images = np.array(procImages)

                # yield a tuple of images and labels
                yield (images, labels)

            # increment the total number of epochs
            epochs += 1

        json_file.close()

    def close(self):
        # close the database
        self.db.close()
