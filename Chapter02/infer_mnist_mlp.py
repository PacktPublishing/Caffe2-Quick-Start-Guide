#!/usr/bin/env python2

"""Compose a MLP network to classify MNIST. Load weights to it and run inference."""

import numpy as np
import operator

from caffe2.python import (
    brew,
    model_helper,
    workspace,
)


# Number of digits in MNIST
MNIST_DIGIT_NUM = 10

# Every grayscale image in MNIST is of dimensions 28x28 pixels in a single channel
MNIST_IMG_HEIGHT = 28
MNIST_IMG_WIDTH = 28
MNIST_IMG_PIXEL_NUM = MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH


def build_mnist_mlp_net(model, input_blob_name):
    """Create a feedforward neural network composed of fullyconnected layers.
    A final softmax layer is used to get probabilities for the 10 numbers."""

    # Create first pair of fullyconnected and ReLU activation layers
    # This FC layer is of size (MNIST_IMG_PIXEL_NUM * 2)
    # On its input side it is fed the MNIST_IMG_PIXEL_NUM pixels
    # On its output side it is connected to a ReLU layer
    fc_layer_0_input_dims = MNIST_IMG_PIXEL_NUM
    fc_layer_0_output_dims = MNIST_IMG_PIXEL_NUM * 2
    fc_layer_0 = brew.fc(
        model,
        input_blob_name,
        "fc_layer_0",
        dim_in=fc_layer_0_input_dims,
        dim_out=fc_layer_0_output_dims
    )
    relu_layer_0 = brew.relu(model, fc_layer_0, "relu_layer_0")

    # Create second pair of fullyconnected and ReLU activation layers
    fc_layer_1_input_dims = fc_layer_0_output_dims
    fc_layer_1_output_dims = MNIST_IMG_PIXEL_NUM * 2
    fc_layer_1 = brew.fc(
        model,
        relu_layer_0,
        "fc_layer_1",
        dim_in=fc_layer_1_input_dims,
        dim_out=fc_layer_1_output_dims
    )
    relu_layer_1 = brew.relu(model, fc_layer_1, "relu_layer_1")

    # Create third pair of fullyconnected and ReLU activation layers
    fc_layer_2_input_dims = fc_layer_1_output_dims
    fc_layer_2_output_dims = MNIST_IMG_PIXEL_NUM
    fc_layer_2 = brew.fc(
        model,
        relu_layer_1,
        "fc_layer_2",
        dim_in=fc_layer_2_input_dims,
        dim_out=fc_layer_2_output_dims
    )
    relu_layer_2 = brew.relu(model, fc_layer_2, "relu_layer_2")

    # Create a softmax layer to provide output probabilities for each of
    # 10 digits. The digit with highest probability value is considered to be
    # the prediction of the network.
    softmax_layer = brew.softmax(model, relu_layer_2, "softmax_layer")

    # Return the last layer of the newly created network
    return softmax_layer


def create_inference_model():

    # Create a model
    inference_model = model_helper.ModelHelper(name="mnist_mlp_model", init_params=False)

    # Build a MLP network in the model
    build_mnist_mlp_net(inference_model, "data")

    return inference_model


def set_model_weights(inference_model):
    """Set the weights of the fully connected layers in the inference network.
    Weights are pre-trained and are read from NumPy files on disk."""

    for i, layer_blob_name in enumerate(inference_model.params):
        layer_weights_filepath = "mnist_mlp_weights/{}.npy".format(str(i))
        layer_weights = np.load(layer_weights_filepath, allow_pickle=False)
        workspace.FeedBlob(layer_blob_name, layer_weights)


def do_inference(inference_model):

    # Read MNIST images from file to use as input
    input_blob = None
    with open("mnist_data.npy") as in_file:
        input_blob = np.load(in_file)

    # Set MNIST image data as input data
    workspace.FeedBlob("data", input_blob)

    # Run inference on the network
    workspace.RunNetOnce(inference_model.net)
    network_output = workspace.FetchBlob("softmax_layer")

    print("Shape of output: ", network_output.shape)

    for i in range(len(network_output)):
        # Quick way to get the top-1 prediction result
        # Squeeze out the unnecessary axis. This returns a 1-D array of length 10
        # Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
        prediction, confidence = max(enumerate(network_output[i]), key=operator.itemgetter(1))
        print("Input: {} Prediction: {} Confidence: {}".format(i, prediction, confidence))


def main():

    # Initialize Caffe2 workspace
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])

    inference_model = create_inference_model()
    set_model_weights(inference_model)
    do_inference(inference_model)


if __name__ == "__main__":
    main()
