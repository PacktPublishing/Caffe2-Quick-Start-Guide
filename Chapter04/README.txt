Follow these steps:

1. Download, build and install Caffe as described in Chapter 4.

2. Download AlexNet using Caffe's download script:

$ cd <caffe directory>
$ python scripts/download_model_binary.py models/bvlc_alexnet/

3. Convert Caffe AlexNet model files to Caffe2 model files using converter script:

$ cd <back to this directory>
$ mkdir bvlc_alexnet
$ python <caffe2 directory>/python/caffe_translator.py \
  <caffe directory>/models/bvlc_alexnet/deploy.prototxt \
  <caffe directory>/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
  --init_net bvlc_alexnet/init_net.pb \
  --predict_net bvlc_alexnet/predict_net.pb

4. Run the AlexNet inference script on any image on CPU:

$ python ch4_alexnet_predictor.py some_image.jpg

5. Run the AlexNet inference script on any image on GPU:

$ python ch4_alexnet_predictor_gpu.py some_image.jpg
