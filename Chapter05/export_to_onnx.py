#!/usr/bin/env python2

"""Script to convert Caffe2 model files to ONNX format.

Input is assumed to be named "data" and of dims (1, 3, 227, 227).
Change INPUT_NAME and INPUT_SHAPE to match that of the network you are
converting.
"""

# Std
import sys

# Ext
import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2

INPUT_NAME = "data"
INPUT_SHAPE = (1, 3, 227, 227)

def main():

    # Check if user provided all required inputs
    if len(sys.argv) != 4:
        print(__doc__)
        print("Usage: " + sys.argv[0] + " <path/to/caffe2/predict_net.pb> <path/to/caffe2/init_net.pb> <path/to/onnx_output.pb>")
        return

    predict_net_fpath = sys.argv[1]
    init_net_fpath = sys.argv[2]
    onnx_model_fpath = sys.argv[3]

    # Read Caffe2 model files to protobuf

    predict_net = caffe2_pb2.NetDef()
    with open(predict_net_fpath, "rb") as f:
        predict_net.ParseFromString(f.read())

    init_net = caffe2_pb2.NetDef()
    with open(init_net_fpath, "rb") as f:
        init_net.ParseFromString(f.read())

    print("Input Caffe2 model name: " + predict_net.name)

    # Network input type, shape and name

    data_type = onnx.TensorProto.FLOAT
    value_info = {INPUT_NAME: (data_type, INPUT_SHAPE)}

    # Convert Caffe2 model protobufs to ONNX

    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
        predict_net,
        init_net,
        value_info,
    )

    # Write ONNX protobuf to file

    print("Writing ONNX model to: " + onnx_model_fpath)
    with open(onnx_model_fpath, "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    main()
