"""
A cnn implementation using tensorflow
To classify pictures of cats and dogs.
"""

import tensorflow as tf
import pandas as pad
import numpy as np
import tflearn
import matplotlib as plt
import random
import os
from tflearn.data_utils import image_preloader
import math
import sys


def get_all_imgs(IMGS_FOLDER = 'dataset'):
  all_imgs = [ 'training_set/cats/'+x for x in os.listdir(IMGS_FOLDER +"/training_set/cats")]
  for dog in os.listdir(IMGS_FOLDER +"/training_set/dogs"):
    all_imgs.append(str('training_set/dogs/'+str(dog)))

  for cat in os.listdir(IMGS_FOLDER +"/test_set/cats"):
    all_imgs.append(str('test_set/cats/'+str(cat)))

  for dog in os.listdir(IMGS_FOLDER +"/test_set/dogs"):
    all_imgs.append(str('test_set/dogs/'+str(dog)))

  random.shuffle(all_imgs)
  return all_imgs


# set the data for training/test/validation
# use the idea of setting the data we will use, but do not load all images at once 
# consumes to much memory
def set_data(all_imgs, TRAIN_DATA = 'training_data.txt', TEST_DATA = 'test_data.txt', VALIDATION = 'validation_data.txt',
             train_proportion = 0.7, test_proportion = 0.2, validation_proportion = 0.1):
  
  IMGS_FOLDER = 'dataset'
  total_imgs = len(all_imgs)

  ######## Training Data ########
  with open(TRAIN_DATA,"w") as file:
    for img in all_imgs[0:int(train_proportion*total_imgs)]:
      if img.find('cat')>=0:
        file.write(IMGS_FOLDER + '/' + img + ' 0\n' )
      elif img.find('dog')>=0:
        file.write(IMGS_FOLDER + '/' + img + ' 1\n' )
      else:
        print("Something is wrong when opening cats/dogs files")
        sys.exit(0)
    file.close()

  ######## Testing Data ########
  with open(TEST_DATA,"w") as file:
    for img in all_imgs[int(math.ceil(train_proportion*total_imgs)):int(math.ceil((train_proportion+test_proportion)*total_imgs))]:
      if img.find('cat')>=0:
        file.write(IMGS_FOLDER + '/' + img + ' 0\n' )
      elif img.find('dog')>=0:
        file.write(IMGS_FOLDER + '/' + img + ' 1\n' )
      else:
        print("Something is wrong when opening cats/dogs files")
        sys.exit(0)
    file.close()

  ######## Validation Data ########
  with open(VALIDATION,"w") as file:
    for img in all_imgs[int(math.ceil((train_proportion+test_proportion)*total_imgs)):total_imgs]:
      if img.find('cat')>=0:
        file.write(IMGS_FOLDER + '/' + img + ' 0\n' )
      elif img.find('dog')>=0:
        file.write(IMGS_FOLDER + '/' + img + ' 1\n' )
      else:
        print("Something is wrong when opening cats/dogs files")
        sys.exit(0)
    file.close()

def main():
  
  all_imgs = get_all_imgs() 
  total_imgs = len(all_imgs)

  set_data(all_imgs)

if __name__ == '__main__':
  main()




