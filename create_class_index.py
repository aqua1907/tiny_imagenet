import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import progressbar

train_path = Path("tiny-imagenet-200/train")
words = pd.read_csv("tiny-imagenet-200/words.txt", delimiter='\t', header=0)

train_labels = os.listdir(train_path)

widgets = ["Dropping rows: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(words),
                               widgets=widgets).start()

for i, row in words.iterrows():
    if row[0] not in train_labels:
        words = words.drop(i)
    pbar.update(i)
pbar.finish()

data = {}

for k, v in enumerate(np.array(words)):
    array = v.tolist()
    array[1] = array[1].split(", ")
    data[k] = array

with open('t_imgNet_class_index.json', 'w') as f:
    json.dump(data, f)
    f.close()
