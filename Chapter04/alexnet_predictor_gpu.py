#!/usr/bin/env python2

"""Determine class of input image using AlexNet model running on GPU.

Script expects init_net.pb and predict_net.pb of BVLC AlexNet to be placed in
the bvlc_alexnet/ directory.

Any input image can be passed in as the first argument to script.
"""

# Std
import PIL.Image
import json
import sys

# Ext
import numpy as np
from caffe2.python import workspace
from caffe2.python import core
from caffe2.proto import caffe2_pb2

GPU_IDX = 0
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

def load_network(device_opt):
    """Create network in workspace on GPU, load weights to GPU."""

    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, "r") as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opt)
        workspace.RunNetOnce(init_def.SerializeToString())

    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, "r") as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opt)
        workspace.CreateNet(net_def.SerializeToString())


def predict_img_class(img, device_opt):
    """Get image class determined by network inference running on GPU."""

    # Copy input image data to GPU
    workspace.FeedBlob("data", img, device_option=device_opt)
    workspace.RunNet("AlexNet", 1)

    # Fetch results of AlexNet's last layer output -- prob
    results = workspace.FetchBlob("prob")

    class_index = np.argmax(results)
    class_prob = results[0, class_index]

    imgnet_classes = json.load(open("imagenet1000.json"))
    class_name = imgnet_classes[class_index]

    return class_index, class_name, class_prob

def main():

    if len(sys.argv) != 2:
        print("Please pass path to input image to this script.")
        return 1

    device_opt = core.DeviceOption(caffe2_pb2.CUDA, GPU_IDX)
    img = prepare_input_image(sys.argv[1])
    load_network(device_opt)
    class_index, class_name, class_prob = predict_img_class(img, device_opt)

    print("Class index: {}, Class: {}, Prob: {}".format(class_index, class_name, class_prob))

if __name__ == "__main__":
    main()
