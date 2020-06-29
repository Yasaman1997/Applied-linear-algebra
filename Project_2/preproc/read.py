import os
from skimage import io
import skimage
from skimage import data  # most functions are in subpackages

# check Pillow version number
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

# 2
import numpy as np
# from PIL import Image
# # Open the image form working directory
# image = Image.open('image.jpg')
# # summarize some details about the image
# print(image.format)
# print(image.size)
# print(image.mode)
# #3

#4
#print(image.mean())

# filename = 'image.jpg'
# #2
# camera = io.imread(filename)
# print(type(camera))
# print("image array:", camera)
#
# #4
# print(camera.mean())
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
import cv2

imnames = ['image.jpg']

# Read collection of images with imread_collection
imlist = (io.imread_collection(imnames))
print(imlist)


