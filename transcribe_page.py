import pandas as pd
import numpy as np

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