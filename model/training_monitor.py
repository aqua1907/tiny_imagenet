import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import numpy as np


# this class is borrowed from the book "Deep Learning for computer vision with python. Practional Bundle", Adrian Rosebrock
class TrainingMonitor(tf.keras.callbacks.BaseLogger):
    def __init__(self, figPath, jsonPath=None, startArt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.starArt = startArt
        self.H = {}

    def on_train_begin(self, logs={}):
        # initialize the history dictionary

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.starArt > 0:
                    # check to see if a starting epoch was supplied
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting epoch
                    for k in self.keys():
                        self.H[k] = self.H[k][:self.starArt]

    def on_epoch_end(self, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            log = self.H.get(k, [])
            log.append(v)
            self.H[k] = log

        if self.jsonPath is not None:
            with open(self.jsonPath, "w") as f:
                json.dump(self.H, f)
                f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="Train Loss")
            plt.plot(N, self.H["val_loss"], label="Val loss")
            plt.plot(N, self.H["accuracy"], label="Train Accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val Accuracy")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # save the figure
            plt.savefig(self.figPath)
            plt.close()
