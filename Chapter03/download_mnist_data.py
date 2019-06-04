#!/usr/bin/env python2

import StringIO
import os
import requests
import zipfile

CAFFE2_MNIST_LMDB_URL = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
DATA_DIR = "mnist_lmdb"


def main():

    if os.path.exists(DATA_DIR):
        print("MNIST data dir already exists: {}".format(DATA_DIR))
        print("Delete the dir if you want to download again.")
        return

    print("Downloading: {}".format(CAFFE2_MNIST_LMDB_URL))
    get_req = requests.get(CAFFE2_MNIST_LMDB_URL, stream=True)
    zfile = zipfile.ZipFile(StringIO.StringIO(get_req.content))

    print("Unzipping to: {}".format(DATA_DIR))
    zfile.extractall(DATA_DIR)


if __name__ == "__main__":
    main()
