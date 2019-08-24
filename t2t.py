import pandas as pd
from transcribe_page import *
from skimage import io
import matplotlib.pyplot as plt


df_train = pd.read_csv('./data/train/train.csv')
imagelist = df_train['Image'].tolist()

image = '200021712-00034_1'

dfs = df_train[df_train['Image'] == image]
'''
for img, labels in dfs.values:
    if type(labels) == float:
        continue
    chars = []
    for unic, x, y, w, h in np.array(labels.split()).reshape(-1, 5):
        chars.append((unic, int(int(x)+int(w)/2), int(int(y)+int(h)/2)))
    print(img)
    print(chars)
    img = io.imread('./data/train/train/{}.jpg'.format(img))
    
    for unic, x, y in chars:
        img[y-10:y+10, x-10:x+10, :] = [255, 0, 0]
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()
    print(sorted([x[1] for x in chars]))
    break
'''
chars = charformat(dfs)

img_shape = io.imread('./data/train/train/{}.jpg'.format(image)).shape

txt = transcribe(chars, img_shape)


    
    