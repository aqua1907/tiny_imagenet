from pathlib import Path


TRAIN_PATH = Path(r"D:\Projects\ML_projects\tiny_imagenet\tiny-imagenet-200\train")
VAL_PATH = Path(r"D:\Projects\ML_projects\tiny_imagenet\tiny-imagenet-200\val\images")
TEST_PATH = Path(r"D:\Projects\ML_projects\tiny_imagenet\tiny-imagenet-200\test\images")
HDF5_PATH = Path(r"D:\Projects\ML_projects\tiny_imagenet\hdf5")


TRAIN_HDF5 = HDF5_PATH/"train.hdf5"
VAL_HDF5 = HDF5_PATH/"val.hdf5"
TEST_HDF5 = HDF5_PATH/"test.hdf5"


DATASET_MEAN = Path(r"D:\Projects\ML_projects\tiny_imagenet\model\tiny_imagenet_mean.json")
SAVED_MODEL_PATH = Path(r"D:\Projects\ML_projects\tiny_imagenet\model\output\tiny_imagenet-200.h5")
JSON_CLASS_INDICES = Path(r'D:\Projects\ML_projects\tiny_imagenet\t_imgNet_class_index.json')
VAL_ANNOTATIONS = Path(r"D:\Projects\ML_projects\tiny_imagenet\tiny-imagenet-200\val\val_annotations.txt")


TENSORBOARD_LOG = Path(r"D:\Projects\ML_projects\tiny_imagenet\model\output")

