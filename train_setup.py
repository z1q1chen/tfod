import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import wget
import tensorflow as tf
import tarfile
import re

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

CHOSEN_MODEL='SSD-ResNet152-V1-FPN-1024'
NUM_STEPS = 40000 #The more steps, the longer the training. Increase if your loss function is still decreasing and validation metrics are increasing. 
NUM_EVAL_STEPS = 500 #Perform evaluation after so many steps

MODELS_CONFIG = {
    # 'efficientdet-d0': {
    #     'model_name': 'efficientdet_d0_coco17_tpu-32',
    #     'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
    #     'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
    #     'batch_size': 16
    # },
    # 'efficientdet-d1': {
    #     'model_name': 'efficientdet_d1_coco17_tpu-32',
    #     'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
    #     'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
    #     'batch_size': 16
    # },
    # 'efficientdet-d2': {
    #     'model_name': 'efficientdet_d2_coco17_tpu-32',
    #     'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
    #     'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
    #     'batch_size': 16
    # },
    #     'efficientdet-d3': {
    #     'model_name': 'efficientdet_d3_coco17_tpu-32',
    #     'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
    #     'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
    #     'batch_size': 16
    # },
    'SSD-MobileNet-V2-320': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-MobileNet-V1-FPN-640': {
        'model_name': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-MobileNet-V2-FPNlite-320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-MobileNet-V2-FPNlite-640': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-ResNet50-V1-FPN-640': {
        'model_name': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-ResNet50-V1-FPN-1024': {
        'model_name': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
        
    'SSD-ResNet101-V1-FPN-640': {
        'model_name': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-ResNet101-V1-FPN-1024': {
        'model_name': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-ResNet152-V1-FPN-640': {
        'model_name': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'SSD-ResNet152-V1-FPN-1024': {
        'model_name': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 16
    }
}

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)

def get_num_classes(pbtxt_fname):
    
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

if __name__ == '__main__':
    test_record_fname = './data/valid/Spaghetti.tfrecord'
    train_record_fname = './data/train/Spaghetti.tfrecord'
    label_map_pbtxt_fname = './data/train/Spaghetti_label_map.pbtxt'

    ##change chosen model to deploy different models available in the TF2 object detection zoo
    

    #in this tutorial we implement the lightweight, smallest state of the art efficientdet model
    #if you want to scale up tot larger efficientdet models you will likely need more compute!
    # chosen_model = 'efficientdet-d0'
    

    

    model_name = MODELS_CONFIG[CHOSEN_MODEL]['model_name']
    pretrained_checkpoint = MODELS_CONFIG[CHOSEN_MODEL]['pretrained_checkpoint']
    base_pipeline_file = MODELS_CONFIG[CHOSEN_MODEL]['base_pipeline_file']
    batch_size = MODELS_CONFIG[CHOSEN_MODEL]['batch_size'] #if you can fit a large batch in memory, it may speed up your training

    save_location = 'research/deploy'
    if not os.path.exists(os.path.join(save_location, pretrained_checkpoint)):
        download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint
        tar_filename = wget.download(download_tar, out=save_location)
        tar = tarfile.open(f'{save_location}/{pretrained_checkpoint}')
        tar.extractall(path=save_location)
        tar.close()
    
    if not os.path.exists(os.path.join(save_location, base_pipeline_file)):
        download_config = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/' + base_pipeline_file
        config_filename = wget.download(download_config, out=save_location)

    #prepare
    pipeline_fname = 'research/deploy/' + base_pipeline_file
    fine_tune_checkpoint = 'research/deploy/' + model_name + '/checkpoint/ckpt-0'
    # num_classes = get_num_classes(label_map_pbtxt_fname)
    num_classes=1

    print('writing custom configuration file')
    with open(pipeline_fname) as f:
        s = f.read()
    with open('pipeline_file.config', 'w') as f:
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                  'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
        
        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                  'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                  'num_steps: {}'.format(NUM_STEPS), s)
        
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                  'num_classes: {}'.format(num_classes), s)
        
        #fine-tune checkpoint type
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
            
        f.write(s)
    print("Done")