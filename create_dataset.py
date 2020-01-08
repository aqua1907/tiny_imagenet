import cv2
import config
import os
from config import tiny_image_net_config as config
from hdf5.hdf5datasetwriter import HDF5Dasetwriter
import numpy as np
import json
import pandas as pd
from imutils import paths
from sklearn.model_selection import train_test_split

(R, G, B) = ([], [], [])


def create_dataset(output, dType):
    with open(config.JSON_CLASS_INDICES) as json_file:
        data = json.load(json_file)

    trainPaths = list(paths.list_images(config.TRAIN_PATH))
    trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]

    (trainPaths, testPaths, trainLabels, testLabels) = train_test_split(trainPaths, trainLabels,
                                                                        test_size=200 * 50,
                                                                        stratify=trainLabels, random_state=9)

    if dType == "train":
        print(f"Build {output}...")
        writer = HDF5Dasetwriter((len(trainPaths), 64, 64, 3), output)

        i = 0
        for (image, label) in zip(trainPaths, trainLabels):
            for k, v in data.items():
                if v[0] == label:
                    img = cv2.imread(image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    (r, g, b) = cv2.mean(img)[:3]
                    R.append(r)
                    G.append(g)
                    B.append(b)

                    writer.add([img], [int(k)])
                    i += 1
                    if i % 10000 == 0:
                        print(f"Added {i} images")

        writer.close()
        json_file.close()

    if dType == "test":
        print(f"Build {output}...")
        writer = HDF5Dasetwriter((len(testPaths), 64, 64, 3), output)

        for (image, label) in zip(testPaths, testLabels):
            for k, v in data.items():
                if v[0] == label:
                    img = cv2.imread(image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    writer.add([img], [int(k)])

        writer.close()

    if dType == "validation":
        print(f"Build {output}...")
        val_annotations = pd.read_csv(config.VAL_ANNOTATIONS, delimiter='\t', names=["IMAGE", "CLASS",
                                                                                     "LEFT", "TOP", "RIGHT",
                                                                                     "BOTTOM"])
        writer = HDF5Dasetwriter((len(val_annotations["IMAGE"]), 64, 64, 3), output)

        for (image, label) in zip(val_annotations["IMAGE"], val_annotations["CLASS"]):
            for k, v in data.items():
                if v[0] == label:
                    img = cv2.imread(str(config.VAL_PATH/image))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    writer.add([img], [int(k)])

        writer.close()


def save_mean_values(path):
    print("Serializing means...")
    tiny_imagenet_mean = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    with open(path, "w") as f:
        f.write(json.dumps(tiny_imagenet_mean))
        f.close()


create_dataset(config.TRAIN_HDF5, dType="train")
create_dataset(config.VAL_HDF5, dType="validation")
create_dataset(config.TEST_HDF5, dType="test")
save_mean_values(config.DATASET_MEAN)





