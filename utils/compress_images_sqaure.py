"""
Script used to resize the scraped images to 256x256 pixels.
The path needs to be updated to the folder that contains the original images
The folder must only contain the images and no other files/directories

Author: Shubham Singhal
Github: shubham21197

Usage: python compress_images_square.py
"""

from PIL import Image
import os

path = './mountain-images-orginal'


for image in os.listdir(path):
    img = Image.open(os.path.join(path, image))
    img = img.resize((256, 256))
    img.save(os.path.join('../data/compressed', image))