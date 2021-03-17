---
title: "Creating ground truth image segmentation maps from VOTT with Python"
date: 2021-03-10T10:48:46-04:00
showDate: true
draft: false
tags: ["blog","story", "python"]
mermaid: true
---
I like VOTT, Microsoft's Open Source Visual Object Tagging Tool - [https://github.com/microsoft/VoTT](https://github.com/microsoft/VoTT), it's easy to use, fast and can handle video too. 
### But how do you create ground truth masks for image segmentation??

Turns out, pretty easily, with a bit of Python...
# The process

{{<mermaid>}}
graph LR
	A[Source images]-->B[VOTT];
	B-->C[Add Boundary];
	C[Python magic] -->D[Groud Truth Masks];
{{</mermaid>}}

``` Python
import os
import cv2
import json
import numpy as np
import shutil

#all your source images -from the export
source_folder = "C:/deeplab/utils/images"
#JSON from the export
json_path = "C:/deeplab/utils/ReceiptBoundary-export.json"                     # Relative to root directory
#The path to multi channel output, the non black and white version
output_masks = "C:/deeplab/utils/results/masksRaw"
#if you want black and white masks they're here
output_maskBW = "C:/deeplab/utils/results/MasksBW"
#The images you've processed so you can keep track
output_images = "C:/deeplab/utils/results/images"

count = 0   
file_map = {}  
MASK_WIDTH = 640 #size of the mask to create
MASK_HEIGHT = 640 #size of the mask to create

# Read JSON file
with open(json_path) as f:
  data = json.load(f)

assets = data['assets']
file_map = {}   
print('\nTotal imags:', len(assets.keys()))



for asset_id in assets.keys():
    asset = assets[asset_id]['asset']
    #this doesn't work for multiple regions, to do that you will need to split by th
    if 'regions' in assets[asset_id].keys():
        regions = assets[asset_id]['regions']
        if len(regions) > 0:
            all_points = []
            for point in regions[0]["points"]:
                all_points.append([point["x"], point["y"]])
            file_map[asset['name']] = all_points
            

            
for itr in file_map:
    to_save_folder = os.path.join(output_images, itr)
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    try:
        arr = np.array(file_map[itr], dtype=np.int32)
    except:
        print("Not found:", itr)
        continue
    count += 1
    #this step is important, you want a colour for each category, this maps to the ID in your label map file
    #to put another if you want to identify car=1, dog=2 then color below would be (1) & then (2) for image
    #yes it can be the same image and overlap
    #If you're using keras segmentation
    cv2.fillPoly(mask, [arr], color=(1))

  
    cv2.imwrite(os.path.join(output_masks, itr.replace('.jpg','.png')) , mask)

    img_grey = cv2.imread(os.path.join(output_masks, itr.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE)


    # threshold the image, 255 will make this black and white
    img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

    #save image
    cv2.imwrite(os.path.join(output_maskBW, itr.replace('.jpg','.png')),img_binary)
    
    #save the original as png in output for good measure
    orig_img = cv2.imread(os.path.join(source_folder, itr))
    png_compression=4
    cv2.imwrite(os.path.join(output_images, itr.replace('.jpg','.png')), orig_img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])

    
    print(itr)
print("Images saved:", count)
```

That's it! 