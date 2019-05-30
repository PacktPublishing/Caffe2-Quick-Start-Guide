Please follow these instructions to run this example:

1. Master branch of pytorch/caffe2 has some problems with reading LMDB files
   to tensors. Please checkout a stable version like v1.0.0, build and use it:

   $ git checkout v1.0.0

2. Download MNIST LMDB train and test datasets provided by Caffe2 using this
   script:

   $ python ./download_mnist_data.py

   You only need to download the datasets once.

3. Train the MNIST LeNet model using this command:

   $ python ./mnist_lenet.py

   During training, the loss and accuracy values are written to loss_accu.csv.
   You can import this file into spreadsheet or plot it to graph.

   The trained model's layer weights files are written to mnist_weights/*.npy
   files. These are the files we used in Chapter 2.

   The trained model is deployed to mnist_model/ dir.

   The script also loads this deployed model and runs an inference on it and
   reports the result.
