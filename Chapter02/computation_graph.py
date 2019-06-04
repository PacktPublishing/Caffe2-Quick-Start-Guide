#!/usr/bin/env python2

"""Create a network that perfoms some mathematical operations.
Run inference on this network."""

from caffe2.python import workspace, model_helper
import numpy as np

# Initialize Caffe2
workspace.GlobalInit(["caffe2",])

# Initialize a model with the name "Math model"
model = model_helper.ModelHelper("Math model")

# Add a matrix multiplication operator to the model.
# This operator takes blobs "A" and "B" as inputs and produces blob "C" as output.
model.net.MatMul(["A", "B"], "C")

# Add a Sigmoid operator to the model.
# This operator takes blob "C" as input and produces blob "D" as output.
model.net.Sigmoid("C", "D")

# Add a Softmax operator to the model.
# This operator takes blob "D" as input and produces blob "E" as output.
model.net.Softmax("D", "E", axis=0)

# Create input A, a 3x3 matrix initialized with some values
A = np.linspace(-0.4, 0.4, num=9, dtype=np.float32).reshape(3, 3)

# Create input B, a 3x1 matrix initialized with some values
B = np.linspace(0.01, 0.03, num=3, dtype=np.float32).reshape(3, 1)

# Feed A and B to the Caffe2 workspace as blobs.
# Provide names "A" and "B" for these blobs.
workspace.FeedBlob("A", A)
workspace.FeedBlob("B", B)

# Run the network inside the Caffe2 workspace.
workspace.RunNetOnce(model.net)

# Extract blob "E" from the workspace.
E = workspace.FetchBlob("E")

# Print inputs A and B and final output E
print "A:", A
print "B:", B
print "E:", E
