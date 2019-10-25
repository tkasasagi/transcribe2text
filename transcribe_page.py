import pandas as pd
import numpy as np
from tqdm import tqdm
import mlcrate as mlc
from skimage import io
from sklearn.cluster import DBSCAN
import random

def charformat(df):    
    charformat = []    
    for index, row in df.iterrows():
        char = row['Unicode']
        x = row['X']
        y = row['Y']
        charformat.append((char, x, y))
    return(charformat)

def transcribe(chars, img_shape):
    # 'chars' in format [(unicode character, x, y), ...] in any order
    density = np.zeros(img_shape[1])
    
    width = img_shape[1] // 50
    for x in [x[1] for x in chars]:
        density[x-width:x+width] += 1
        
    columns = []
    col = None
    for ptr in range(len(density)):
        height = density[ptr]
        if col is None and height > 0:
            col = ptr
        if col and height == 0:
            columns.append((col, ptr, []))
            col = None
            
    chars = sorted(chars, key=lambda x: x[2])
    for char, x, y in chars:
        for i, (left, right, _) in enumerate(columns):
            if x < right:
                columns[i][2].append((char, x, y))
                break
                
    output = ''
    for _, _, chars in columns[::-1]:
        for unicode, _, _ in chars:
            char = chr(int(unicode[2:], 16))
            output += char
        output += '\n'

    return output.strip()

def t2c(img, labels):    
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
        
    xlst = np.array(xlst).astype('float32')
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(xlst)
    
    distances, indices = nbrs.kneighbors(xlst)
    
    mean_dist = np.mean(distances)
    print("MEAN NEAREST NEIGHBOR DISTANCE", np.mean(distances))
    
    clustering = DBSCAN(eps= mean_dist * 4.0, min_samples=1)

    
    #xlst[:,0] *= 0.1
    
    xlst_cluster = xlst * 1.0
    
    xlst_cluster[:,0] *= 5.0
    
    print('xlst shape', xlst.shape)
    
    #print('xlst', xlst)
    
    cluster_labels = clustering.fit_predict(xlst_cluster)
        
        
    #pd.Series(cluster_labels).value_counts()
    
    print('cl', cluster_labels)
    
    cluster2chars = {}
    
    for j in range(0, len(cluster_labels)):
        #print('cluster', cluster_labels[j])
        #print('char', xlst[j])
        
        if not cluster_labels[j] in cluster2chars: 
            cluster2chars[cluster_labels[j]] = []

        cluster2chars[cluster_labels[j]].append(xlst[j])
            
    cluster2color = {}
            
    for keyc in cluster2chars: 
        #print(keyc)
        randcolor = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        #print(cluster2chars[keyc])
        cluster2color[keyc] = randcolor
        for chararray in cluster2chars[keyc]:
            x = int(chararray[0])
            y = int(chararray[1])
            img[y-10:y+10, x-10:x+10, :] = randcolor
        
        
        
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
    
    img[0 : 40*(len(clusters)) + 10, 0:50, :] = 255
    for j in range(len(clusters)):
        img[(j)*40 + 10 : (j)*40 + 40, 0 : 50, :] = cluster2color[clusters[j]]
            
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.show()