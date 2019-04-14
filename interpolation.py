# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

from functools import partial
import tensorflow as tf
import cv2

from time import sleep
import serial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--face1', type=str, default='./dlatents/face1.npy')
parser.add_argument('--face2', type=str, default='./dlatents/face2.npy')
parser.add_argument('--visualize_images', dest='save_images', action='store_false')
parser.add_argument('--generate_video', type=str, default='False')
parser.add_argument('--num_images', type=int, default=80)
parser.set_defaults(save_images = True)

args = parser.parse_args()
print(args)

# Initialize TensorFlow.
tflib.init_tf()
print("Loading the model...")
# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

# Print network details.
# Gs.print_layers()

counter = 0
stop_while = False

# Load the latents of each face
dlatent1 = np.load(args.face1)
dlatent1 = dlatent1.repeat(18, axis=2)[0]

dlatent2 = np.load(args.face2)
dlatent2 = dlatent2.repeat(18, axis=2)[0]

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

while not stop_while:
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    alphas = np.linspace(0,1,args.num_images/2) # Initial value - Final value - Num of interpolations

    for a in alphas:
        dlatent = (1-a)*dlatent1 + a*dlatent2
        image = Gs.components.synthesis.run(dlatent, randomize_noise=False, **synthesis_kwargs)

        if args.save_images:
            # Save image.
            os.makedirs(config.result_dir, exist_ok=True)
            png_filename = os.path.join(config.result_dir, '{}-{}-{}.png'.format(args.face1[:-4], args.face2[:-4], counter))
            PIL.Image.fromarray(image[0], 'RGB').save(png_filename)
            print("Image '{}-{}-{}.png' saved".format(args.face1[:-4], args.face2[:-4], counter), end='\r')
        else:
            # Display the image with opencv
            im = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
            cv2.imshow('', im)
            cv2.waitKey(1)
        
        counter += 1
        if counter > args.num_images:
            stop_while = True
            break

    aux = dlatent1
    dlatent1 = dlatent2
    dlatent2 = aux

if args.generate_video:
    image_folder = args.result_dir
    video_name =  '{}-{}.avi'.format(args.face1[:-4], args.face2[:-4])

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    order = np.asarray([int(img.split('_')[-1].split('.')[0]) for img in images])

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 27, (width,height))

    for i in range(len(images)):
        video.write(cv2.imread(os.path.join(image_folder, images[np.where(order==i)[0][0]])))

    cv2.destroyAllWindows()
    video.release()

print("Succes! :D")

