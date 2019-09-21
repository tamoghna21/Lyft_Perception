# Standard imports
import os
import re
import pickle
import sys
import time

from glob import glob
from math import ceil
from urllib import request
from zipfile import ZipFile
import tarfile

# Third-party imports
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
import imageio
import cv2
from skimage import transform


# Install tqdm
#!pip install tqdm
from tqdm import tqdm

print('Python', sys.version)
print('Numpy', np.__version__)
print('Tensorflow', tf.__version__)

A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = A4_PORTRAIT[::-1]

IMAGE_ROAD_TOP = 275
IMAGE_ROAD_BOTTOM = 525
IMAGE_BOTTOM_HOOD = 496



PIXEL_ROAD = 7
PIXEL_CAR = 10
PIXEL_MAX = 255



# Download Lyft challenge training data
data_url='https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz'
	
zip_fn = '../lyft_data/lyft_training_data.tar.gz'
dest = '../lyft_data/'
if not os.path.exists(dest+'Train'):
	print("entering")
	if not os.path.exists(zip_fn): #Expecting the zip file at ../lyft_data location
		request.urlretrieve(data_url, zip_fn)
	if (zip_fn.endswith("tar.gz")):	
		tar = tarfile.open(zip_fn, "r:gz")
		tar.extractall(dest)
		tar.close()
	elif (zip_fn.endswith("tar")):	
		tar = tarfile.open(zip_fn, "r:")
		tar.extractall(dest)
		tar.close()
		

# Download pretrained VGG16
vgg_url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip'
vgg_fn = '../lyft_data/vgg16.zip'
vgg_path = '../lyft_data/'
if not os.path.exists(vgg_path+'vgg'):
	print("etering")
	if not os.path.exists(vgg_fn): #Expecting the zip file vgg16.zip at ../lyft_data
		request.urlretrieve(vgg_url, vgg_fn)
	
	vgg_zip = ZipFile(vgg_fn)
	vgg_zip.extractall(vgg_path)

print("End 1st part")
input()


######

def augment_brightness_camera_images(image):
  # Randomly increase or decrease brightness, not used
  image = colors.rgb_to_hsv(image).astype(float)
  random_bright = .5 + np.random.uniform()
  image[:, :, 2] = image[:, :, 2] * random_bright
  image[:, :, 2][image[:, :, 2] > PIXEL_MAX] = PIXEL_MAX
  image = colors.hsv_to_rgb(image).astype(int)
  return image

def rotate(image, label, angle):
  # Rotate by given angle
  angle = np.random.random() * angle * 2 - angle
  image = scipy.misc.imrotate(image, angle)
  label = scipy.misc.imrotate(label, angle)
  return image, label

def flip(image, label):
  # Flip images horizontally
  image = image[:, ::-1, :]
  label = label[:, ::-1, :]
  return image, label

def randomize(image, label):
    # Randomize a single image
    image = augment_brightness_camera_images(image)
    image, label = rotate(image, label, angle=5)
    if np.random.random() < .5:
        image, label = flip(image, label)
    return image, label


def gen_batch(feature_img_paths, image_shape, augment=False):
    # Returns a batch generator for given images
    def get_batches(batch_size, shuffle=False):
        if shuffle:
            np.random.shuffle(feature_img_paths)
        foreground_road = np.array([PIXEL_ROAD, 0, 0])
        foreground_car = np.array([PIXEL_CAR, 0, 0])
        for i in range(0, len(feature_img_paths), batch_size):
            features, labels = [], []
            for fn in feature_img_paths[i:i + batch_size]:
                # Preprocess features
                feature = imageio.imread(fn)
                #feature = scipy.misc.imread(fn)
                
                feature = scipy.misc.imresize(feature, image_shape) # If scipy is used, image_shape = (160, 576)
                #feature = cv2.resize(feature, image_shape) # If cv2 is used, image_shape = (576, 160)
                #feature = transform.resize(feature, image_shape)
		
		
		        # Load labels
                label = imageio.imread(fn.replace('CameraRGB', 'CameraSeg'))
		        #label = scipy.misc.imread(fn.replace('CameraRGB', 'CameraSeg'))
		
		        #***************************************************************
		        ##The following steps marks the HOOD pixels as Other(0)
                a = label[IMAGE_BOTTOM_HOOD:,:,:]==10
                b=np.copy(label[IMAGE_BOTTOM_HOOD:,:,:])
                b[a]=0
                label=np.concatenate((label[:IMAGE_BOTTOM_HOOD,:,:],b),axis=0)
		        #***************************************************************
		
		        #***************************************************************
		        #The following step marks the lane marks as roads
                label[label==6]=7
		        #***************************************************************
                
        
                label = scipy.misc.imresize(label, image_shape)
		        #label = cv2.resize(label, image_shape)
        
                if augment:
                    feature, label = randomize(feature, label)

                # Convert label image to output array
                gt_fg_road = np.all(label == foreground_road, axis=2)
                gt_fg_car = np.all(label == foreground_car, axis=2)
                gt_road_and_car = gt_fg_road | gt_fg_car

                gt_fg_road = gt_fg_road.reshape(*gt_fg_road.shape, 1)
                gt_fg_car = gt_fg_car.reshape(*gt_fg_car.shape, 1)
                gt_road_and_car = gt_road_and_car.reshape(*gt_road_and_car.shape, 1)

                label = np.concatenate((gt_fg_road,gt_fg_car,np.invert(gt_road_and_car)), axis=2)
		
		
                features.append(feature)
                labels.append(label)
            yield np.array(features), np.array(labels)

    return get_batches
print("End of 2nd cell")
input()
  
####################
data_dir = '../lyft_data/Train'
data_dir1 = '../lyft_data/carla-capture-20180513A'

image_shape = (160, 576)
#image_shape = (576, 160)

######################
##To check how many images have road pixels above 275 and below 525

print("No code here: skipping this part")
print("End of 3rd cell Done")
input()

####################
#Load VGG-16 model and build FCN

num_classes = 3

# Load saved model
vgg_tag = 'vgg16'

with tf.Session() as sess:
  tf.saved_model.loader.load(sess, [vgg_tag], vgg_path + 'vgg')

# Gather required tensor references
vgg_input_tensor_name = 'image_input:0'
vgg_keep_prob_tensor_name = 'keep_prob:0'
vgg_layer3_out_tensor_name = 'layer3_out:0'
vgg_layer4_out_tensor_name = 'layer4_out:0'
vgg_layer7_out_tensor_name = 'layer7_out:0'

graph = tf.get_default_graph()
input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

# Regularizers and initializers
# initializer = lambda: tf.contrib.layers.xavier_initializer()
initializer = lambda: tf.truncated_normal_initializer(stddev=0.01)
regularizer = lambda: tf.contrib.layers.l2_regularizer(1e-5)

# 1x1 convolution
layer7_out = tf.layers.conv2d(
	inputs=vgg_layer7_out,
	filters=num_classes,
	kernel_size=1,
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer())

# Upsample
layer7_up = tf.layers.conv2d_transpose(
	inputs=layer7_out,
	filters=num_classes,
	kernel_size=4,
	strides=(2, 2),
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer())

# 1x1 convolution
layer4_out = tf.layers.conv2d(
	inputs=vgg_layer4_out,
	filters=num_classes,
	kernel_size=1,
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer())

# Skip layer
skip_1 = tf.add(layer7_up, layer4_out)

# Upsample
skip_1_up = tf.layers.conv2d_transpose(
	inputs=skip_1,
	filters=num_classes,
	kernel_size=4,
	strides=(2, 2),
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer())

# 1x1 convolution
layer3_out = tf.layers.conv2d(
	inputs=vgg_layer3_out,
	filters=num_classes,
	kernel_size=1,
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer())

# Skip layer
skip_2 = tf.add(skip_1_up, layer3_out)

# Upsampled final
nn_last_layer = tf.layers.conv2d_transpose(
	inputs=skip_2,
	filters=num_classes,
	kernel_size=16,
	strides=(8, 8),
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer())

print("Done")
print("End of 4th cell Done")
input()
##################################

#Define Training operations
correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],name = 'correct_label')
learning_rate = tf.placeholder(tf.float32)
logits = tf.reshape(nn_last_layer, (-1, num_classes))
predicts = tf.nn.softmax(logits, name='predicts')
truth = tf.reshape(correct_label, (-1, num_classes))
global_step = tf.Variable(0, name='global_step', trainable=False)

# Cross-entropy operation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=truth)
cross_entropy_loss = tf.reduce_mean(cross_entropy, name='loss')
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

# Regularization loss; this is a good thing

l2_loss = tf.losses.get_regularization_loss()
#l2_loss = tf.losses.get_regularization_losses()
tf.summary.scalar('l2_loss', l2_loss)
total_loss = cross_entropy_loss + l2_loss
#total_loss = cross_entropy_loss 
tf.summary.scalar('total_loss', total_loss)

# Exponential decaying learning rate
'''
min_lr = 5e-5
edlr = tf.train.exponential_decay(
    learning_rate=learning_rate,
    global_step=global_step,
    decay_steps=1,
    decay_rate=0.99)
edlr = tf.maximum(edlr, min_lr)  # Prevent underflow and updates getting too small
# edlr = learning_rate  # Disable exponential decay
tf.summary.scalar('learning_rate', edlr)
train_op = tf.train.AdamOptimizer(learning_rate=edlr).minimize(total_loss, global_step)
'''
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step)

# Merge summary operation
merged = tf.summary.merge_all()
print("Done")
print("End of 5th cell Done")
input()
####################################

#Perform training if no saved model is available

# Hyperparameters
augment_training = True
batch_size = 16
epochs = 50
min_epochs = 10
patience = 3
random_seed = 42
starting_learning_rate = 1e-4

# Training configuration


saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('tensorboard')  # , sess.graph) Disable graph
    print("here")
    sess.run(tf.global_variables_initializer())
    print("here1")
	# Path to vgg model
	#vgg_path = os.path.join('models', 'vgg')'../lyft_data/'

	# Prepare training and test datasets
    filenames = glob(os.path.join(data_dir, 'CameraRGB', '*.png'))
    filenames.extend(glob(os.path.join(data_dir1, 'CameraRGB', '*.png')))
    print('Dataset contains', len(filenames), 'examples.')
    np.random.seed(random_seed)  # Make it reproduceable
    np.random.shuffle(filenames)
    training = int(len(filenames) * 1.0)  # 85% for training, 15% for testing
    training, testing = filenames[:training], filenames[training:]
	
    pickle.dump(testing, open('testing_data_pickle_bothv4.p',"wb"))
    print("Test data saved in pickle")

	# Check a saved model exists or train a new one
    if os.path.exists('./model_combinedv4.ckpt.meta'):
	  # assert False  # Force training a new model
	  #assert os.path.exists('./model_combined.ckpt.meta')
        print("model already exists")
    else:
	  # Train new model
        print('Training on', len(training), 'samples.')
        get_batches_fn = gen_batch(
		  feature_img_paths=training,
		  image_shape=image_shape,
		  augment=augment_training)

        # Continue training while the loss improves
        best_loss = 1e9
        failure = 0
        for e in range(epochs):
            epoch_loss = 0
            num_images = 0
            sys.stdout.flush()
            for images, labels in tqdm(get_batches_fn(batch_size, shuffle=True),
								   desc='Training epoch {}'.format(e + 1),
								   total=ceil(len(training) / batch_size)):
			    #print(labels.dtype)
                summary, step, _, loss = sess.run([
			        merged,
			        global_step,
			        train_op,
			        cross_entropy_loss], feed_dict={
				    input_image: images,
				    correct_label: labels,
				    keep_prob: 0.75,
				    learning_rate: starting_learning_rate})
                writer.add_summary(summary, step)
                epoch_loss += loss * len(images)
                num_images += len(images)

            epoch_loss /= num_images
            sys.stderr.flush()
            print('Epoch {} loss: {:.3f}'.format(e + 1, epoch_loss))
            print(num_images)
            if e >= min_epochs and epoch_loss > best_loss:
                if failure == patience:
                    break
                failure += 1
            else:
                failure = 0
                best_loss = epoch_loss
                print('Saving model')
                saver.save(sess, './model_combinedv4.ckpt')
print("Training Done")
print("End of 6th cell Done")

########################
