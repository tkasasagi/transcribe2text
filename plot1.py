import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transcribe_page import *
from skimage import io
from os import listdir
from PIL import Image, ImageDraw, ImageFont
image_size = 960

file = 'wo01_03959_0001_p0005-0'
df = pd.read_csv('./decoder/csv/{}.csv'.format(file))

#Draw result on image
imsource = Image.open('./decoder/source/{}.jpg'.format(file)).convert('RGBA')

width, height = imsource.size
tmp = Image.new('RGBA', imsource.size)

draw = ImageDraw.Draw(tmp)

fx = []
fy = []
xs = df['x']
ys = df['y']
cha = df['char']

rx = width/image_size
ry = height/image_size
for x, y, chax in zip(xs, ys, cha):
    #character = charmap[chax]          
    color = 'rgb(255, 0, 0)'
    xz = x * rx #calculate back to original size (the model recognize in 640x640 size)
    yz = y * ry #calculate back to original size (the model recognize in 640x640 size)
    fx.append(xz)
    fy.append(yz)
    xb = xz 
    yb = yz    
    draw.rectangle((xb, yb, xb + 1 , yb + 1), fill=(255, 0, 0, 255))
    #draw.text((xb, yb - fontsize/3), character, fill=color, font = font)
imsource = Image.alpha_composite(imsource, tmp)
imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

imsource.save('./decoder/ocr/{}.jpg'.format(file)) #save image with ocr result
print('OCR image saved')