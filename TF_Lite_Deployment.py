
# Description: 
# This program uses a TensorFlow Lite object detection model to perform object 
# detection on an image or a folder full of images. It draws boxes and scores 
# around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py


# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from pyzbar.pyzbar import decode

# Imports for sensors
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2, Preview
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import sys
import hx711 as HX711

# Set-up defaults for sensors 
sys.stdin.flush()
sys.stdout.flush()
#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)

#set GPIO Pins
GPIO_TRIGGER = [6, 21]
GPIO_ECHO = [5, 20]

Time_Sleep = 0.0001
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
for i in range(len(GPIO_ECHO)):
    GPIO.setup(GPIO_ECHO[i], GPIO.IN)
    
# Define functions for Sensors
CAMERA_SAVE_PATH = os.path.join(os.getcwd(),"Images")

def image(file_name = "test"):
    time.sleep(0.1)
    image_path = os.path.join(CAMERA_SAVE_PATH, (file_name + ".jpg"))
    picam2.capture_file(image_path)

def distance(all_distances = True, echo_idx = 0):
    if (all_distances):
        
        StartTime = [0]*len(GPIO_ECHO)
        StopTime = [0]*len(GPIO_ECHO)
        TimeElapsed = [0]*len(GPIO_ECHO)
        distance = [0]*len(GPIO_ECHO)
        
        for idx in range(len(GPIO_ECHO)):
            # set Trigger to HIGH
            GPIO.output(GPIO_TRIGGER[idx], True)
 
            # set Trigger after 0.01ms to LOW
            time.sleep(Time_Sleep)
            GPIO.output(GPIO_TRIGGER[idx], False)
            
            StartTime[idx] = time.time()
            StopTime[idx] = time.time()
            
            # Define Timeout clock
            TimeOutClk = time.time()
            
            # save StartTime
            while GPIO.input(GPIO_ECHO[idx]) == 0:
                StartTime[idx] = time.time()
                if ((StartTime[idx] - TimeOutClk) > 0.1):
                    break
 
            # save time of arrival
            while GPIO.input(GPIO_ECHO[idx]) == 1:
                StopTime[idx] = time.time()
                if ((StopTime[idx] - StartTime[idx]) > 0.1):
                    break
 
            # time difference between start and arrival
            TimeElapsed[idx] = StopTime[idx] - StartTime[idx]
            # multiply with the sonic speed (34300 cm/s)
            # and divide by 2, because there and back
            distance[idx] = (TimeElapsed[idx] * 34300) / 2
            
    else:
        # set Trigger to HIGH
        GPIO.output(GPIO_TRIGGER[echo_idx], True)
 
        # set Trigger after 0.01ms to LOW
        time.sleep(Time_Sleep)
        GPIO.output(GPIO_TRIGGER[echo_idx], False)
        
        StartTime = time.time()
        StopTime = time.time()
        
        # save StartTime
        while GPIO.input(GPIO_ECHO[echo_idx]) == 0:
            StartTime = time.time()

        # save time of arrival
        while GPIO.input(GPIO_ECHO[echo_idx]) == 1:
            StopTime = time.time()
 
        # time difference between start and arrival
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2   
 
    return distance

######################################################
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default= "TFLite_Barcode_Model")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default="Images")
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder',
                    action='store_true', default=True)
parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)',
                    action='store_true')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

save_results = args.save_results # Defaults to False
show_results = args.noshow_results # Defaults to True

IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'test1.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
    if save_results:
        RESULTS_DIR = IM_DIR + '_results'

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)
    if save_results:
        RESULTS_DIR = 'results'

# Create results directory if user wants to save results
if save_results:
    RESULTS_DIR_PATH = os.path.join(CWD_PATH,RESULTS_DIR)
    BOUNDED_DIR_PATH = RESULTS_DIR_PATH + '/Bounded'
    if not os.path.exists(RESULTS_DIR_PATH):
        os.makedirs(RESULTS_DIR_PATH)
        os.makedirs(BOUNDED_DIR_PATH)
        
"""    
#Delete all results
for file_name in os.listdir(RESULTS_PATH):
    # construct full file path
    file = os.path.join(RESULTS_PATH,file_name)
    if not os.path.isdir(file):
        os.remove(file)

for file_name in os.listdir(BOUNDED_DIR_PATH):
    # construct full file path
    file = os.path.join(BOUNDED_DIR_PATH,file_name)
    if not os.path.isdir(file):
        os.remove(file)
"""

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

def detect_barcode():
    # Loop over every image and perform detection
    for image_path in images:

        # Load image and resize to expected shape [1xHxWx3]
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        detections = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
                
                
                #Save bounded barcode
                bounded_filename = "Barcode{}_".format(i) + os.path.basename(image_path)
                bounded_savepath = os.path.join(BOUNDED_DIR_PATH,bounded_filename)
                bounded_image = image[ymin:ymax, xmin:xmax]
                cv2.imwrite(bounded_savepath, bounded_image)
                cv2.imshow('bounded', bounded_image)
                cv2.waitKey(0)

        # All the results have been drawn on the image, now display the image
        if show_results:
            cv2.imshow('Object detector', image)
            
            #Press any key to continue to next image, or press 'q' to quit
            if cv2.waitKey(0) == ord('q'):
               break

        # Save the labeled image to results folder if desired
        if save_results:
            
            # Get filenames and paths
            image_fn = os.path.basename(image_path)
            image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
            
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn +'.txt'
            txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_result_fn)

            # Save image
            cv2.imwrite(image_savepath, image)

            # Write results to text file
            # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            with open(txt_savepath,'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
      
    # Clean up
    cv2.destroyAllWindows()

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
            
            
            #Save bounded barcode
            bounded_filename = "Barcode{}_".format(i) + os.path.basename(image_path)
            bounded_savepath = os.path.join(BOUNDED_DIR_PATH,bounded_filename)
            bounded_image = image[ymin:ymax, xmin:xmax]
            cv2.imwrite(bounded_savepath, bounded_image)
            cv2.imshow('bounded', bounded_image)
            cv2.waitKey(0)

        # All the results have been drawn on the image, now display the image
        if show_results:
            cv2.imshow('Object detector', image)
            
            #Press any key to continue to next image, or press 'q' to quit
            if cv2.waitKey(0) == ord('q'):
               break

        # Save the labeled image to results folder if desired
        if save_results:
            
            # Get filenames and paths
            image_fn = os.path.basename(image_path)
            image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
            
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn +'.txt'
            txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_result_fn)

            # Save image
            cv2.imwrite(image_savepath, image)

            # Write results to text file
            # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            with open(txt_savepath,'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
      
    # Clean up
    cv2.destroyAllWindows()

    for filename in os.listdir(BOUNDED_DIR_PATH):
        #script_dir = os.path.dirname(os.path.abspath(__file__))
            barcode_path = os.path.join(BOUNDED_DIR_PATH, filename)
            barcode = cv2.imread(barcode_path, cv2.IMREAD_COLOR)
            if image is None:
                print("Image is empty or None.")
            else:
                # Try to read the barcode directly without transforming the image
                barcode_data = read_barcode(barcode)
                #barcode_data = None
                if barcode_data is None:
                    # If barcode is not detected, try detecting the boundary and transforming the image
                    square_boundary = auto_detect_boundary(barcode)
                    if square_boundary is not None:
                        warped_image = four_point_transform(barcode, square_boundary)
                        barcode_data = read_barcode(warped_image)
                    else:
                        print("No square boundary detected.")

######################################################

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def read_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)

    if barcodes:
        barcode_data = barcodes[0].data.decode("utf-8")
        print(f"Barcode data: {barcode_data}")
        return barcode_data
    else:
        print("No barcode detected")
        return None

def auto_detect_boundary(image):
    if image is None:
        print("Image is empty or None.")
        return None
    print('inside')
    print(f"Image shape: {image.shape}")  # Add this line to check the image shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Add this line to blur the image
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None

if __name__ == "__main__":
    try:
        # Set-up camera 
        picam2 = Picamera2()
        camera_config = picam2.create_preview_configuration()
        picam2.configure(camera_config)
        picam2.start_preview(Preview.QTGL)
        picam2.start()
        
        # Set-up detection variables 
        current_detection_status = 0
        previous_detection_status = 0
        threshold = 10
        
        while True:
            if len(GPIO_ECHO) == 1: # If one ultrasonic sensor
                dist = distance()
                print(dist)
                
            elif len(GPIO_ECHO) == 2: # If two ultrasonic sensors
                """
                direction = "NONE"
                dist = distance()
                if ((dist[0]-dist[1]) >= 50):
                    direction ="LEFT"
                elif ((dist[0]-dist[1]) <= -50):
                    direction = "RIGHT"
                else:
                    direction = "NONE"
                
                print("Measured Distance (0,1): %.1f cm, %.1f cm, %s" % (dist[0], dist[1], direction))
                # for idx in range(len(dist)):
                #    print("Measured Distance for sensor %.1f = %.1f cm" % (idx , dist[idx]))
                #print("Measured Distance = %.1f cm" % dist)
                """
                
                dist = distance()
                
                previous_detection_status = current_detection_status
                
                if (dist[0] < threshold) or (dist[1] < threshold):
                    current_detection_status = 1
                else:
                    current_detection_status = 0
                
                if (current_detection_status == 1) and (previous_detection_status == 0):
                    print("Hand going into cart")
                    image(file_name = "hand_in")
                    detect_barcode()
                elif (current_detection_status == 0) and (previous_detection_status == 1):
                    print("Hand leaving cart")
                    image(file_name = "hand_out")
                    detect_barcode()
                    
                print('.')
                time.sleep(0.1)
 
                # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
