#!/usr/bin/env python2

# Std
import PIL.Image
import json
import sys

# Ext
import numpy as np
from caffe2.python import workspace

import onnx
import caffe2.python.onnx.backend

# # Prepare the inputs, here we use numpy to generate some random inputs for demo purpose
# import numpy as np
# rand_input = np.random.randn(1, 3, 224, 224).astype(np.float32)


IMG_SIZE = 227
MEAN = 128

def prepare_input_image(img_fpath):
    """Read and prepare input image as AlexNet input."""

    # Read input image as 3-channel 8-bit values
    pil_img = PIL.Image.open(img_fpath)

    # Resize to AlexNet input size
    res_img = pil_img.resize((IMG_SIZE, IMG_SIZE), PIL.Image.LANCZOS)

    # Convert to NumPy array and float values
    img = np.array(res_img, dtype=np.float32)

    # Change HWC to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)

    # Change RGB to BGR
    img = img[(2, 1, 0), :, :]

    # Mean subtraction
    img = img - MEAN

    # Change CHW to NCHW by adding batch dimension at front
    img = img[np.newaxis, :, :, :]

    return img

def predict_img_class(onnx_model, img):
    """Get image class determined by network."""

    results = caffe2.python.onnx.backend.run_model(onnx_model, [img])
    class_index = np.argmax(results[0])
    class_prob = results[0][0, class_index]

    imgnet_classes = json.load(open("imagenet1000.json"))
    class_name = imgnet_classes[class_index]

    return class_index, class_name, class_prob

def main():

    if len(sys.argv) != 3:
        print("Invoke this script as: python ch5_run_onnx_model.py <path to alexnet.onnx> <path to image file>")
        return 1

    model = onnx.load(sys.argv[1])
    img = prepare_input_image(sys.argv[2])
    class_index, class_name, class_prob = predict_img_class(model, img)

    print("Class index: {}, Class: {}, Prob: {}".format(class_index, class_name, class_prob))

if __name__ == "__main__":
    main()
