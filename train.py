import matplotlib
from config import tiny_image_net_config as config
from hdf5.hdf5datasetgenerator import HDF5DatasetGenerator
from model.RestNet50 import ResNet
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time

matplotlib.use("Agg")

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode='nearest')

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug, binarize=False, classes=200, mean=True)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, binarize=False, classes=200, mean=True)

print("Compiling model..")

opt = tf.keras.optimizers.Adam()

model = ResNet.build(64, 64, 3, 200, (3, 4, 6), (64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")

# early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5,
#                                               verbose=1, restore_best_weights=True)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_LOG, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, min_lr=0.00001, patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint(config.SAVED_MODEL_PATH, best_only=True)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


start_time = time.time()

history = model.fit_generator(trainGen.generator(),
                              steps_per_epoch=trainGen.numImages // 64,
                              validation_data=valGen.generator(),
                              validation_steps=valGen.numImages // 64,
                              epochs=75,
                              max_queue_size=64*2,
                              callbacks=[tensorboard_callback])

print("Serializing model...")
model.save(config.SAVED_MODEL_PATH)

finish_time = time.time()
print("Took time: ", time.strftime("%H:%M:%S", time.gmtime(finish_time - start_time)))

trainGen.close()
valGen.close()
