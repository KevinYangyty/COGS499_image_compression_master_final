# modified from source https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/train.caltech.py#L27

import os
import pandas as pd
from params import TrainingParams

import numpy as np

tparam = TrainingParams(verbose=False)

image_dir_list         = os.listdir( tparam.images_path )
#image_dir_list.remove('.DS_Store')
print(image_dir_list)
label_pairs            = list(map(lambda x: x.split('.'), image_dir_list))
print(label_pairs)
labels, label_names    = zip(*label_pairs)

labels                 = list(map(lambda x: int(x), labels))


label_dict             = pd.Series(labels, index = label_names) - 1
image_paths_per_label  = list(map(lambda one_dir: list(map(lambda one_file: os.path.join( tparam.images_path, one_dir, one_file ), os.listdir( os.path.join( tparam.images_path, one_dir)))), image_dir_list))
image_paths_train      = np.hstack(list(map(lambda one_class: one_class[:-10], image_paths_per_label)))

image_paths_test       = np.hstack(list(map(lambda one_class: one_class[-10:], image_paths_per_label)))

trainset               = pd.DataFrame({'image_path': image_paths_train})
testset                = pd.DataFrame({'image_path': image_paths_test })

trainset               = trainset[ trainset['image_path'].map( lambda x: x.endswith('.jpg'))]
trainset['label']      = trainset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
trainset['label_name'] = trainset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

testset                = testset[ testset['image_path'].map( lambda x: x.endswith('.jpg'))]
testset['label']       = testset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
testset['label_name']  = testset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])


if not os.path.exists(tparam.data_train_path.split('/')[0]):
    os.makedirs(tparam.data_train_path.split('/')[0])

trainset.to_pickle(tparam.data_train_path)
testset.to_pickle(tparam.data_test_path)
