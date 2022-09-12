from fastapi import FastAPI, File, UploadFile
from utils import load_style_image, load_content_image, show_n
import tensorflow as tf
import tensorflow_hub as hub
import os
import matplotlib.pyplot as plt
from fastapi import Response
import json
from PIL import Image

app = FastAPI()

def style_img_path(style):
    images = os.listdir('./img')
    if style in images:
        img_path = os.path.join('img',style)
    return img_path

def generate(file,style):
    #image sizes
    content_img_size = (300,300)
    style_img_size = (256, 256) 
    
    print("loading content image")
    content_image = load_content_image(file, content_img_size)
    print("loading style image")
    style_image = load_style_image(style, style_img_size)
    style_image = tf.nn.avg_pool2d(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
    print("images shape: ",content_image.shape, style_image.shape)
    #show_n([content_image, style_image], ['Content image', 'Style image'])
    with tf.device('/cpu:0'):
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        hub_module = hub.load(hub_handle)

        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
        #show_n([stylized_image],['result'])
    return stylized_image
    
@app.post("/")
def file_process(file: UploadFile, style_name:str):
    result = generate(file.file,style_img_path(style_name))[0].numpy()
    plt.imshow(result,aspect='equal')
    plt.show()
    plt.imsave("test1.jpg",result)
    return Response(json.dumps(result.tolist()))
