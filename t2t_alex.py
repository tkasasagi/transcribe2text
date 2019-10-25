import pandas as pd
import numpy as np
from skimage import io
from sklearn.cluster import DBSCAN
import random
import matplotlib.pyplot as plt

FONTPATH = 'NotoSansCJKjp-Regular.otf'
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype(FONTPATH, 70, encoding='utf-8')

df_train = pd.read_csv('kaggle_train.csv')

filename = '200021763-00023_1'
img = df_train[df_train['image_id'] == filename].iloc[0,0]
labels = df_train[df_train['image_id'] == filename].iloc[0,1]

pimage = Image.open('./data/train/train/' + img + ".jpg")
pdraw = ImageDraw.Draw(pimage)
#for img, labels in df_train.values:
    
xlst = []

if type(labels) == float:
    exit
chars = []
for unic, x, y, w, h in np.array(labels.split()).reshape(-1, 5):
    chars.append((unic, int(int(x)+int(w)/2), int(int(y)+int(h)/2)))

img = io.imread('./data/train/train/{}.jpg'.format(img))

for unic, x, y in chars:
    img[y-10:y+10, x-10:x+10, :] = [255, 0, 0]
    xlst.append([x,y])
    pdraw.text((x, y), '1', fill='rgb(255,0,0)', font = font)
    
#pimage.show()    
pimage.save('result.jpg')
    
xlst = np.array(xlst).astype('float32')

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(xlst)

distances, indices = nbrs.kneighbors(xlst)

mean_dist = np.mean(distances)
print("MEAN NEAREST NEIGHBOR DISTANCE", np.mean(distances))

clustering = DBSCAN(eps= mean_dist * 4.0, min_samples=1)

xlst_cluster = xlst * 1.0

xlst_cluster[:,0] *= 5.0

print('xlst shape', xlst.shape)

cluster_labels = clustering.fit_predict(xlst_cluster)
    
print('cl', cluster_labels)

cluster2chars = {}

for j in range(0, len(cluster_labels)):   
    if not cluster_labels[j] in cluster2chars: 
        cluster2chars[cluster_labels[j]] = []

    cluster2chars[cluster_labels[j]].append(xlst[j])


cluster2color = {}
        
for keyc in cluster2chars: 
    randcolor = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    cluster2color[keyc] = randcolor
    for chararray in cluster2chars[keyc]:
        x = int(chararray[0])
        y = int(chararray[1])
        img[y-10:y+10, x-10:x+10, :] = randcolor

plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.show()    

clusters = list(cluster2chars.keys())

print('clusters', clusters)
#Reorder the clusters!
#Steps: 
#Check for neighbor below bottom.  If it's close add it.  
#Check for neighbor near tops.  If it's close add it.  
#Otherwise take the right-most column.  

new_clusters = []

while True:
    
    #Break if all clusters are assigned.  
    if len(clusters) == 0:
        break
  
    #Below-logic
    
    if len(new_clusters) > 0:
        last_chars = np.array(cluster2chars[new_clusters[-1]])
        
        last_bottom_y = last_chars[:,1].max()
        last_bottom_x = last_chars[:,0].mean()
        
        smallest_dist = float('inf')
        closest = -1
        
        for cluster in clusters:
            chars = np.array(cluster2chars[cluster])
            
            bottom_y = chars[:,1].min()
            bottom_x = chars[:,0].mean()
            
            dist = abs(last_bottom_y - bottom_y) + 5.0 * abs(last_bottom_x - bottom_x)
            
            if dist < smallest_dist and bottom_y > last_bottom_y:
                smallest_dist = dist
                closest = cluster
                
        #print('smallest_dist', smallest_dist, 'threshold', mean_dist*7.0)
        if smallest_dist < mean_dist * 7.0: 
            clusters.remove(closest)
            new_clusters.append(closest)
            print('Taking new cluster by neighbors!', closest)
    
    if len(clusters) == 0:
        break
    
    #Right-most logic.  
    rightmost = -float('inf')
    pick = -1
    for cluster in clusters:
        chars = np.array(cluster2chars[cluster])
        hor_center = chars[:,0].mean() - 0.1 * chars[:,1].mean()
        if hor_center > rightmost: 
            pick = cluster
            rightmost = hor_center
            
    print('taking by rightmost logic', pick)
    clusters.remove(pick)
    new_clusters.append(pick)

clusters = new_clusters

print('new clusters', clusters)

#ordering cluster
img[0 : 40*(len(clusters)) + 10, 0:50, :] = 255
for j in range(len(clusters)):
    img[(j)*40 + 10 : (j)*40 + 40, 0 : 50, :] = cluster2color[clusters[j]]
        
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.show()
