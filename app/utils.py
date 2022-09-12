import functools
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_style_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  img = tf.io.decode_image(
      tf.io.read_file(image_url),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  print('**********************',type(img),'******',img.shape)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

@functools.lru_cache(maxsize=None)
def load_content_image(image, image_size=(256, 256), preserve_aspect_ratio=True):
  img = np.array(Image.open(image),dtype=np.float32)/255.0
  img = np.expand_dims(img,0)
  img = tf.convert_to_tensor(img)
  print('**********************',type(img),img.shape)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()