#!/usr/bin/env python2

"""Train a MNIST LeNet model and use it for inference."""

import os
import shutil
import numpy as np

import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import (
    brew,
    core,
    model_helper,
    optimizer,
    workspace,
)

DATA_DIR = "mnist_lmdb"
MODEL_DIR = "mnist_model"
WEIGHTS_DIR = "mnist_weights"

def del_create_dir(dpath):
    """If dir exists, delete it.
    Then create dir."""

    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.makedirs(dpath)

def add_model_inputs(model, batch_size, db, db_type):

    # Load data from DB
    input_images_uint8, input_labels = brew.db_input(
        model,
        blobs_out=["input_images_uint8", "input_labels"],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )

    # Cast grayscale pixel values to float
    # Scale pixel values to [0, 1]
    input_images = model.Cast(input_images_uint8, "input_images", to=core.DataType.FLOAT)
    input_images = model.Scale(input_images, input_images, scale=float(1./256))

    # We do not need gradient for backward pass
    # This op stops gradient computation through it
    input_images = model.StopGradient(input_images, input_images)

    return input_images, input_labels

def build_mnist_lenet(model, input_blob_name):
    """Build the LeNet network for MNIST."""

    # Convolution layer that operates on the input MNIST image
    # Input is grayscale image of size 28x28 pixels
    # After convolution by 20 kernels each of size 5x5,
    # output is 20 channels, each of size 24x24
    layer_1_input_dims = 1   # Input to layer is grayscale, so 1 channel
    layer_1_output_dims = 20 # Output from this layer has 20 channels
    layer_1_kernel_dims = 5  # Each kernel is of size 1x5x5
    layer_1_conv = brew.conv(
        model,
        input_blob_name,
        "layer_1_conv",
        dim_in=layer_1_input_dims,
        dim_out=layer_1_output_dims,
        kernel=layer_1_kernel_dims,
    )

    # Max-pooling layer that operates on output from previous convolution layer
    # Input is 20 channels, each of size 24x24
    # After pooling by 2x2 windows and stride of 2, the output of this layer
    # is 20 channels, each of size 12x12
    layer_2_kernel_dims = 2 # Max-pool over 2x2 windows
    layer_2_stride = 2      # Stride by 2 pixels between each pool
    layer_2_pool = brew.max_pool(
        model,
        layer_1_conv,
        "layer_2_pool",
        kernel=layer_2_kernel_dims,
        stride=layer_2_stride,
    )

    # Convolution layer that operates on output from previous pooling layer.
    # Input is 20 channels, each of size 12x12
    # After convolution by 50 kernels, each of size 20x5x5,
    # the output is 50 channels, each of size 8x8
    layer_3_input_dims = 20  # Number of input channels
    layer_3_output_dims = 50 # Number of output channels
    layer_3_kernel_dims = 5  # Each kernel is of size 50x5x5
    layer_3_conv = brew.conv(
        model,
        layer_2_pool,
        "layer_3_conv",
        dim_in=layer_3_input_dims,
        dim_out=layer_3_output_dims,
        kernel=layer_3_kernel_dims,
    )

    # Max-pooling layer that operates on output from previous convolution layer
    # Input is 50 channels, each of size 8x8
    # Apply pooling by 2x2 windows and stride of 2
    # Output is 50 channels, each of size 4x4
    layer_4_kernel_dims = 2 # Max-pool over 2x2 windows
    layer_4_stride = 2      # Stride by 2 pixels between each pool
    layer_4_pool = brew.max_pool(
        model,
        layer_3_conv,
        "layer_4_pool",
        kernel=layer_4_kernel_dims,
        stride=layer_4_stride,
    )

    # Fully-connected layer that operates on output from previous pooling layer
    # Input is 50 channels, each of size 4x4
    # Output is vector of size 500
    layer_5_input_dims = 50 * 4 * 4
    layer_5_output_dims = 500
    layer_5_fc = brew.fc(
        model,
        layer_4_pool,
        "layer_5_fc",
        dim_in=layer_5_input_dims,
        dim_out=layer_5_output_dims,
    )

    # ReLU layer that operates on output from previous fully-connected layer
    # Input and output are both of size 500
    layer_6_relu = brew.relu(
        model,
        layer_5_fc,
        "layer_6_relu",
    )

    # Fully-connected layer that operates on output from previous ReLU layer
    # Input is of size 500
    # Output is of size 10, the number of classes in MNIST dataset
    layer_7_input_dims = 500
    layer_7_output_dims = 10
    layer_7_fc = brew.fc(
        model,
        layer_6_relu,
        "layer_7_fc",
        dim_in=layer_7_input_dims,
        dim_out=layer_7_output_dims,
    )

    # Softmax layer that operates on output from previous fully-connected layer
    # Input and output are both of size 10
    # Each output (0 to 9) is a probability score on that digit
    layer_8_softmax = brew.softmax(
        model,
        layer_7_fc,
        "softmax",
    )

    return layer_8_softmax


def add_accuracy_op(model, softmax_layer, label):
    """Adds an accuracy op to the model"""
    brew.accuracy(model, [softmax_layer, label], "accuracy")


def add_logging_ops(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """

    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     MODEL_DIR/[blob name]
    model.Print("accuracy", [], to_file=1)
    model.Print("loss", [], to_file=1)

    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)


def create_train_model(data_folder):
    """Create model for training with MNIST train dataset."""

    # Create the model helper for the train model
    train_model = model_helper.ModelHelper(name="mnist_lenet_train_model")

    # Specify the input is from the train lmdb
    data, label = add_model_inputs(
        train_model,
        batch_size=64,
        db=os.path.join(data_folder, "mnist-train-nchw-lmdb"),
        db_type="lmdb",
    )

    # Build the LeNet-5 network
    softmax_layer = build_mnist_lenet(train_model, data)

    # Compute cross entropy between softmax scores and labels
    cross_entropy = train_model.LabelCrossEntropy([softmax_layer, label], "cross_entropy")

    # Compute the expected loss
    loss = train_model.AveragedLoss(cross_entropy, "loss")

    # Use the average loss we just computed to add gradient operators to the model
    train_model.AddGradientOperators([loss])

    # Specify the optimization algorithm
    optimizer.build_sgd(
        train_model,
        base_learning_rate=0.1,
        policy="step",
        stepsize=1,
        gamma=0.999,
    )

    # Track the accuracy of the model
    add_accuracy_op(train_model, softmax_layer, label)

    return train_model

def create_test_model(data_folder):
    """Create model for testing using MNIST test dataset."""

    test_model = model_helper.ModelHelper(name="mnist_lenet_test_model", init_params=False)
    data, label = add_model_inputs(
        test_model,
        batch_size=100,
        db=os.path.join(data_folder, "mnist-test-nchw-lmdb"),
        db_type="lmdb",
    )
    softmax_layer = build_mnist_lenet(test_model, data)
    add_accuracy_op(test_model, softmax_layer, label)

    return test_model

def create_deploy_model(data_folder):
    """Create model for deployment."""

    deploy_model = model_helper.ModelHelper(name="mnist_lenet_deploy_model", init_params=False)
    build_mnist_lenet(deploy_model, "input_images")
    return deploy_model

def create_train_test_models(data_folder):
    """Create train, test and deployment models.

    We do not use test model in this example. But you can use it with
    validation datasets for your own training.
    """

    train_model = create_train_model(data_folder)
    test_model = create_test_model(data_folder)
    deploy_model = create_deploy_model(data_folder)
    return train_model, test_model, deploy_model

def do_training(train_model):
    """Train the MNIST model."""

    # Init parameter blobs and create network.
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # Training values
    total_iters = 100
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)

    # Iterate and train
    for i in range(total_iters):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.blobs["accuracy"]
        loss[i] = workspace.blobs["loss"]
        print("Iteration: {}, Loss: {}, Accuracy: {}".format(i, loss[i], accuracy[i]))

    # Write loss and accuracy to CSV file.
    # This can be imported to a spreadsheet or plotted to a graph.
    loss_accu_fpath = "loss_accu.csv"
    print("Writing loss and accuracy to: ", loss_accu_fpath)
    with open(loss_accu_fpath, "w") as loss_file:
        for l, a in zip(loss, accuracy):
            loss_file.write(str(l) + "," + str(a) + "\n")

def save_model(deploy_model):
    """Write layer weights to files. These are used in Chapter 2.
    Also, write model to file.
    """

    del_create_dir(WEIGHTS_DIR)

    # Iterate weights of model
    for i, blob in enumerate(deploy_model.params):
        blob_vals = workspace.FetchBlob(blob)
        wfpath = "{}/{}.npy".format(WEIGHTS_DIR, str(i))

        # Write weights to file.
        # These are the weights we imported to our model in Chapter 2.
        print("Writing weights file:", wfpath)
        with open(wfpath, "w") as ofile:
            np.save(ofile, blob_vals, allow_pickle=False)

    # Create model for export.
    # Specify input and output blobs.
    pe_meta = pe.PredictorExportMeta(
        predict_net=deploy_model.net.Proto(),
        parameters=[str(b) for b in deploy_model.params],
        inputs=["input_images"],
        outputs=["softmax"],
    )

    # Save model to file in minidb format.
    del_create_dir(MODEL_DIR)
    pe.save_to_db("minidb", os.path.join(MODEL_DIR, "mnist_model.minidb"), pe_meta)
    print("Deploy model written to: " + MODEL_DIR + "/mnist_model.minidb")

def do_inference():
    """Run inference on the deployed model."""

    # Extract input images blob from workspace
    input_blob = workspace.FetchBlob("input_images")

    # Reset workspace for inference.
    workspace.ResetWorkspace(MODEL_DIR)

    # Load model saved to file.
    predict_net = pe.prepare_prediction_net(os.path.join(MODEL_DIR, "mnist_model.minidb"), "minidb")

    # Feed input for inference.
    workspace.FeedBlob("input_images", input_blob)

    # Do inference.
    workspace.RunNetOnce(predict_net)
    softmax = workspace.FetchBlob("softmax")

    # Get the prediction.
    pred_digit = np.argmax(softmax[0])
    confidence = softmax[0][pred_digit]
    print("Predicted digit: ", pred_digit)
    print("Confidence: ", confidence)


def main():
    core.GlobalInit(["caffe2", "--caffe2_log_level=0"])
    train_model, test_model, deploy_model = create_train_test_models(DATA_DIR)
    do_training(train_model)
    save_model(deploy_model)
    do_inference()


if __name__ == "__main__":
    main()
