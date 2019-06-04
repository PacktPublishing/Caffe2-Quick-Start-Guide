#!/usr/bin/env python2

"""Determine class of input image using AlexNet model.

Script expects init_net.pb and predict_net.pb of BVLC AlexNet to be placed in
the bvlc_alexnet directory.

Any input image can be passed in as the first argument to script.
"""

# Std
import PIL.Image
import json
import sys

# Ext
import numpy as np
from caffe2.python import workspace

IMG_SIZE = 227
MEAN = 128
INIT_NET = "bvlc_alexnet/init_net.pb"
PREDICT_NET = "bvlc_alexnet/predict_net.pb"

def prepare_input_image(img_fpath):
    """Read and prepare input image as AlexNet input."""

    # Read input image as 3-channel 8-bit values
    pil_img = PIL.Image.open(sys.argv[1])

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

def load_network():
    """Create network, load weights and create a predictor from it."""

    with open(INIT_NET) as f:
        init_net = f.read()
    with open(PREDICT_NET) as f:
        predict_net = f.read()

    predictor = workspace.Predictor(init_net, predict_net)
    return predictor

def predict_img_class(predictor, img):
    """Get image class determined by network."""

    results = predictor.run({"data": img})
    class_index = np.argmax(results[0])
    class_prob = results[0][0, class_index]

    imgnet_classes = json.load(open("imagenet1000.json"))
    class_name = imgnet_classes[class_index]

    return class_index, class_name, class_prob

def main():

    if len(sys.argv) != 2:
        print("Please pass path to input image to this script.")
        return 1

    img = prepare_input_image(sys.argv[1])
    predictor = load_network()
    class_index, class_name, class_prob = predict_img_class(predictor, img)

    print("Class index: {}, Class: {}, Prob: {}".format(class_index, class_name, class_prob))

if __name__ == "__main__":
    main()
