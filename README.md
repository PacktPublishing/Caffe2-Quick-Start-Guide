# Caffe2-Quick-Start-Guide
Published by Packt

<a href="Packt UTM URL of the Book"><img src="Cover Image URL of the Book" alt="Book Name" height="256px" align="right"></a>

This is the code repository for [Caffe2 Quick Start Guide](Packt UTM URL of the Book), published by Packt.

**Modular and scalable deep learning made easy**

## What is this book about?
Caffe2 is a popular deep learning library used for fast and scalable training and inference of deep learning models on various platforms. This book introduces you to the Caffe2 framework and shows how you can leverage its power to build, train, and deploy efficient neural network models at scale.

This book covers the following exciting features: 
* Build and install Caffe2
* Compose neural networks
* Train neural network on CPU or GPU
* Import a neural network from Caffe
* Import deep learning models from other frameworks

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1789137756) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
# Cast grayscale pixel values to float
# Scale pixel values to [0, 1]
input_images = model.Cast(input_images_uint8, "input_images",
to=core.DataType.FLOAT)
input_images = model.Scale(input_images, input_images, scale=float(1./256))
```

**Following is what you need for this book:**
Data scientists and machine learning engineers who wish to create fast and scalable deep learning models in Caffe2 will find this book to be very useful. Some understanding of the basic machine learning concepts and prior exposure to programming languages like C++ and Python will be useful.	

With the following software and hardware list you can run all code files present in the book (Chapter 01-07).

### Software and Hardware List

| Chapter  | Software required                   | OS required                        |
| -------- | ------------------------------------| -----------------------------------|
| 1        | R version 3.3.0                     | Windows, Mac OS X, and Linux (Any) |
| 2        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 3        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 4        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 5        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 6        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 7        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 8        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 9        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 10        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 11        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 12        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 13        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 14        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |
| 15        | Rstudio Desktop 0.99.903            | Windows, Mac OS X, and Linux (Any) |


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](Graphics Bundle Link).

## Code in Action

Click on the following link to see the Code in Action:

[Placeholder link](www.youtube.com/URL)

### Related products <Other books you may enjoy>
* [Deep Learning with PyTorch Quick Start Guide](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch-quick-start-guide?utm_source=github&utm_medium=repository&utm_campaign=9781789534092) [[Amazon]](https://www.amazon.com/dp/1789534097)

* [Deep Learning Quick Reference](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-quick-reference?utm_source=github&utm_medium=repository&utm_campaign=9781788837996) [[Amazon]](https://www.amazon.com/dp/1788837991)

## Get to Know the Author
**Ashwin Nanjappa**
is a senior architect at NVIDIA, working in the TensorRT team on improving deep learning inference on GPU accelerators. He has a PhD from the National University of Singapore in developing GPU algorithms for the fundamental computational geometry problem of 3D Delaunay triangulation. As a post-doctoral research fellow at the BioInformatics Institute (Singapore), he developed GPU-accelerated machine learning algorithms for pose estimation using depth cameras. As an algorithms research engineer at Visenze (Singapore), he implemented computer vision algorithm pipelines in C++, developed a training framework built upon Caffe in Python, and trained deep learning models for some of the world's most popular online shopping portals.

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
