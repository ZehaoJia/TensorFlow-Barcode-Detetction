

# Automatic Checkout with Tensorflow Object Detection 
## Summary 
This project entails using machine learning to automatically detect and extract barcodes from a live video feed in order to facilitate a seamless automatic checkout system. At the core of this system is an object detection model trained using Tensor Flow, and deployed as a Tensorflow Lite model to the Raspberry Pi. An overview of the project workflow is illustrated below:

(Overview)

The following sections will document the approach to the project in chronological order. Note that only core script and training data are included in this repo as the trained models are too large to include. Tensorflow setup and model training are based on this [guide](https://github.com/nicknochnack/TFODCourse).

## 1. Setup 

**Virtual Environment and Dependencies:**
 A python virtual environment is created and linked to a Jupyter Notebook to compartmentalize project dependencies. Using Jupyter interactive environment, the required dependencies for image collection and annotation are installed. This part is completed with *Image_Collection.ipynb*.

**Collecting and Annotating images:** 
In preparation for model training, a collection of images with barcodes is required. For this, a mix of sample images from Kaggle, as well as hand-taken images are used. These images are then annotated to indicate the barcode location in them. This part is completed with *Image_Collection.ipynb* and the annotated data can be found at 

## 2. Training
**Tensorflow Setup and Enable GPU:**
Tensorflow and its required dependencies are installed with *Image_Collection.ipynb*. GPU training is enabled by following the instructions on the official Tensorflow [website](https://www.tensorflow.org/install/source_windowsinstalling).

**Model Training:**
The annotated data is converted to TFrecords(required for Tensorflow training) and the model is trained with *Model_Training.ipynb*. The specifications of the model trained is listed below:

(spec)

## 3. Deployment
**Export to Tensorflow Lite:**
The trained model is converted to a Tensorflow Lite model, a light weight version that is more suitable for mobile devices. This step is completed with *Model_Training.ipynb*.

**Integrate into Raspberry Pi System:**
The light weight model is integrated in a functioning prototype that is able to detect the barcode of products being placed into a basket. This prototype is composed of a Raspberry Pi with a camera and ultrasonic sensors embedded into a laser cut crate. 

(Prototype image)

A general overview of the prototype function can be described as:
1. Ultrasonic sensors detect the motion of products being placed into the basket.
2. The camera is triggered to take snapshots of the product from various angles.
3. The collected images are passed into the detection model.
4. If any barcodes are detected, they are bound and extracted. 
5. The extracted barcode is decoded and the item is checked out.

The python script *TF_Lite_Deployment.py* is used to facilitate the listed functionality. 

## 4. Evaluation

**Training Evaluation:**
The following figures illustrate the evaluation on model training.
(Eval1)

**Deployment Results:**
The following figures illustrate the results from prototype. 
(Eval2)