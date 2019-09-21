import sys
import os
from copy import deepcopy
from glob import glob

import numpy as np
import tensorflow as tf

def test_safe(func):
    """
    Isolate tests
    """
    def func_wrapper(*args):
        with tf.Graph().as_default():
            result = func(*args)
        print('Tests Passed')
        return result

    return func_wrapper

@test_safe
def test_for_training_dataset(data_dir):
    #training_dataset_path = os.path.join(data_dir, 'data_road')
    training_dataset_path = data_dir
    training_labels_count = len(glob(os.path.join(training_dataset_path, 'Train/CameraRGB/*.png')))
    training_images_count = len(glob(os.path.join(training_dataset_path, 'Train/CameraSeg/*.png')))
    #testing_images_count = len(glob(os.path.join(training_dataset_path, 'testing/image_2/*.png')))

    assert not (training_images_count == training_labels_count == 0),\
        'Training dataset not found. Extract Training dataset in {}'.format(kitti_dataset_path)
    assert training_images_count == 1000, 'Expected 1000 training images, found {} images.'.format(training_images_count)
    assert training_labels_count == 1000, 'Expected 1000 training labels, found {} labels.'.format(training_labels_count)
    #assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)