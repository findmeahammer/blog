---
title: "Custom object detection on Windows - Pascal VOC to TF Lite Model Maker"
date: 2021-05-30T18:48:46-04:00
showDate: true
draft: false
tags: ["blog","story"]
mermaid: false
---

# Custom object detection on Windows -Pascal VOC to TF Lite Model Maker

Create custom training data using VoTT, export it to Pascal VOC. Then convert it into TFRecords to be consumed by TF Lite Model Maker to train your custom model.

This is a more thorough guide than available on [https://www.tensorflow.org/lite/tutorials/model_maker_object_detection](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection), using the new tools released at Google IO 2021.

For the jupyter notebook of this see: [https://github.com/findmeahammer/ml/blob/65783a1febf828adb159851d91ef27b450be16df/tflite_modelMaker.ipynb](https://github.com/findmeahammer/ml/blob/65783a1febf828adb159851d91ef27b450be16df/tflite_modelMaker.ipynb)


### Requirements
The below all need to be installed
- nvidia GPU setup CUDA toolkit, see the guide here: [https://www.tensorflow.org/install/gpu#windows_setup](https://www.tensorflow.org/install/gpu#windows_setup)
- Tensorflow 2 installed, follow the guide here: [https://www.tensorflow.org/install/pip#windows](https://www.tensorflow.org/install/pip#windows)
- Python 3.7 seems to be the best
- Jupyter notebook/lab

Strongly recommend a Python environment manager e.g. Anaconda. However, the TF Lite install is currently only available on PIP. 

The below example are using Anaconda with Jupyter Lab.

## Testing the Tensorflow installation

### 1. Activate the Python environment where Tensorflow is installed
`conda activate tf2`
### 2. Start jupyter
`juyter lab`

### 3. Test tensorflow
``` python
import tensorflow as tf;
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```

You should see output similar to: `tf.Tensor(1672.2386, shape=(), dtype=float32)` if not then there's something wrong with your install. Check your environment/path variables. 



## Installing TF Lite Model Maker
### 1. Install pycocotools
`!pip install -q pycocotools`

### 2. Install model maker. 

`pip install tflite-model-maker`
If you encounter an error try the nightly:
`pip install tflite-model-maker-nightly`

### 3. Test the installation
Run the below to make sure model maker is working, try nightly if you enounter an error

``` python 
import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR);
```

## 4. Create training data
Download VoTT (or another tool that can export Pascal VOC) [https://github.com/microsoft/VoTT](https://github.com/microsoft/VoTT)
Install, follow the guide on github and annotate some images:
- create new project, set your source/destination folders
- add some images to the source folder
- on the far right tool window add a new annotation
- Select an image and highlight the area you want to identify, press the number key on the keyboard (e.g. 1) for that particular annotation.
- Repeat for lots of images -few hundred ideally although 50 or so may be ok to test the theory.
Once you have annotted your images check your destination/export settings.
- You want Pascal VOTT format
- Tagged images only
- Don't worry about grouping

Now press export in the top menu bar (arrow pointing out). 

To make life easier for yourself setup a directory structure like:
- C:\tflite\test1\
-   c:\tflite\test1\annotations
-   c:\tflite\test1\images
-   c:\tflite\test1\export
-   c:\tflite\test1\tfrecord

Now copy the exported images from vott into the images folder above. Do the same for XML annotation files. 

The below snippet will create the other directories for you (and delete the content of any existing directories!!)

``` python 
train_root_dir = 'C:/tfw/tflite/may20/'

import os 
import shutil

validate_images = os.path.join(train_root_dir, 'validate/images')

shutil.rmtree(os.path.join(train_root_dir, 'validate/images'))
os.makedirs(os.path.join(train_root_dir, 'validate/images'))
shutil.rmtree(os.path.join(train_root_dir, 'validate/annotations'))
os.makedirs(os.path.join(train_root_dir, 'validate/annotations'))

shutil.rmtree(os.path.join(train_root_dir, 'train/images'))
os.makedirs(os.path.join(train_root_dir, 'train/images'))
shutil.rmtree(os.path.join(train_root_dir, 'train/annotations'))
os.makedirs(os.path.join(train_root_dir, 'train/annotations'))

shutil.rmtree(os.path.join(train_root_dir, 'test/images'))
os.makedirs(os.path.join(train_root_dir, 'test/images'))
shutil.rmtree(os.path.join(train_root_dir, 'test/annotations'))
os.makedirs(os.path.join(train_root_dir, 'test/annotations'))
```

#### Create train/eval/test datasets
The below will take the source images and xml and split them into training, evaluation, and test folders:
``` python 
import os 
import random

source_images = os.path.join(train_root_dir, 'images')
source_annotations = os.path.join(train_root_dir, 'annotations')

random.seed();

images_list = os.listdir(source_images)

random.shuffle(images_list)
print(len(images_list))

total_images = len(images_list)    #total images in your source
train_count = int(total_images * 0.85) #training size 85% of total
validation_count = int(total_images * 0.1) #validation size, 10% of total
test_count = total_images -(train_count + validation_count)  #test size, 5/6% of total depending on rounding
print(train_count)
print(validation_count)
print(test_count)

for file in images_list[:train_count]:
    #print(file)
    source_file = os.path.join(source_images,file)
    #copy images
    #print(source_file)
    #print(os.path.join(train_root_dir, 'train/images',file))
    shutil.copyfile(source_file, os.path.join(train_root_dir, 'train/images',file))
    #copy annotation
    annotation_file = os.path.splitext(file)[0] + '.xml'
    shutil.copyfile(os.path.join(source_annotations,annotation_file), os.path.join(train_root_dir, 'train/annotations',annotation_file))

for file in images_list[train_count:(train_count + validation_count)]:
    #print(file)
    source_file = os.path.join(source_images,file)
    #copy images
    #print(source_file)
    #print(os.path.join(train_root_dir, 'train/images',file))
    shutil.copyfile(source_file, os.path.join(train_root_dir, 'validate/images',file))
    #copy annotation
    annotation_file = os.path.splitext(file)[0] + '.xml'
    shutil.copyfile(os.path.join(source_annotations,annotation_file), os.path.join(train_root_dir, 'validate/annotations',annotation_file))
    
for file in images_list[(train_count + validation_count):]:
    #print(file)
    source_file = os.path.join(source_images,file)
    #copy images
    #print(source_file)
    #print(os.path.join(train_root_dir, 'train/images',file))
    shutil.copyfile(source_file, os.path.join(train_root_dir, 'test/images',file))
    #copy annotation
    annotation_file = os.path.splitext(file)[0] + '.xml'
    shutil.copyfile(os.path.join(source_annotations,annotation_file), os.path.join(train_root_dir, 'test/annotations',annotation_file))
```
    
### Create tfrecords from Pascal VOC
Now it's time to create the tfrecords which will be used by the model maker

First step is to create the label defintion
`label_map={1: "dog"}`
The dog/number should be replaced with the annotation you created in VoTT. You can also copy/paste from the pbtxt file in the export from VoTT folder.

The below will generate tfrecords for train, validation and test
``` python 
###train

image_dir = os.path.join(train_root_dir, 'train/images')
annotations_dir =os.path.join(train_root_dir, 'train/annotations')
output_train =os.path.join(train_root_dir,'tfrecord/train')

max_images = 300
object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map=label_map, cache_dir=output_train, max_num_images=max_images)
train_images = len(os.listdir(image_dir))

###validate
image_dir = os.path.join(train_root_dir, 'validate/images')
annotations_dir =os.path.join(train_root_dir, 'validate/annotations')
output_dir =os.path.join(train_root_dir,'tfrecord/validate')
max_images = 100
object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map=label_map, cache_dir=output_dir, max_num_images=max_images)
validate_images = len(os.listdir(image_dir))

###test
image_dir = os.path.join(train_root_dir, 'test/images')
annotations_dir =os.path.join(train_root_dir, 'test/annotations')
output_dir =os.path.join(train_root_dir,'tfrecord/test')
max_images = 100
object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map=label_map, cache_dir=output_dir, max_num_images=max_images)
test_images = len(os.listdir(image_dir))
```

### Load the data ready for training

``` python
data_set_size =train_images
train_data = object_detector.DataLoader(
    "C:/tfw/tflite/may20/tfrecord/train/*.tfrecord", data_set_size, label_map=label_map, annotations_json_file=None
)

data_set_size =validate_images
validation_data = object_detector.DataLoader(
    "C:/tfw/tflite/may20/tfrecord/validate/*.tfrecord", data_set_size, label_map=label_map, annotations_json_file=None
)

data_set_size =test_images
test_data = object_detector.DataLoader(
    "C:/tfw/tflite/may20/tfrecord/test/*.tfrecord", data_set_size, label_map=label_map, annotations_json_file=None
)
```

## Training
Load a prebuilt model to run inferrence on

`spec = model_spec.get('efficientdet_lite0')`
See the tensorflow guides for more information/othre models

### Train!
`model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)`
You will now see something like:
Epoch 1/50
33/33 [==============================] - 53s 351ms/step - det_loss: 1.5920 - cls_loss: 1.0355 - box_loss: 0.0111 - reg_l2_loss: 0.0630 - loss: 1.6550 - learning_rate: 0.0090 - gradient_norm: 2.0824 - val_det_loss: 1.3866 - val_cls_loss: 0.6407 - val_box_loss: 0.0149 - val_reg_l2_loss: 0.0630 - val_loss: 1.4496
Epoch 2/50

Depending on your GPU this may take some time....

### Evaluation the mode
`model.evaluate(test_data)`
The closer the 'AP' is to 1 the better your model is at identifying the object. Note if you have poor training/test data this may be better than real world. If the AP is 1 then there's a good chance you've got overfit. Get more data.

### Export the model

To export to the tensorflow saved model format:

`out_dir ='C:/tflite/test1/export'
model.export(export_dir=out_dir, export_format=[ExportFormat.SAVED_MODEL, ExportFormat.LABEL])`

To export to TFLite to be used on a mobile device:
`model.export(export_dir=out_dir, tflite_filename='may25model.tflite', export_format=ExportFormat.TFLITE)`

## Testing the model

``` python
import cv2

from PIL import Image

model_path = out_dir + '/may25model.tflite'

# Load the labels into a list
classes = ['???'] * model.model_spec.config.num_classes
label_map = model.model_spec.config.label_map
for label_id, label_name in label_map.as_dict().items():
  classes[label_id-1] = label_name

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  
  # Get all outputs from the model
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8
```
Specify a test image and run.
``` python
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import os
import glob


TEMP_FILE = 'C:/test/eval/PXL_20210304_134057864.jpg' #REPLACE WITH A TEST IMAGE
DETECTION_THRESHOLD = 0.3


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

#for image_path in TEST_IMAGE_PATHS:
    # Run inference and draw detection result on the local copy of the original file
detection_result_image = run_odt_and_draw_results(
    TEMP_FILE,
    interpreter,
    threshold=DETECTION_THRESHOLD
)

# Show the detection result
Image.fromarray(detection_result_image)
```

That's it. the model.lite can now be imported into an Android/iOS/JS project to do realtime object detection using your custom data set.
