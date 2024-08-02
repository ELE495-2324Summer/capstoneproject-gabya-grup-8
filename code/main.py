from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# Store the message to be retrieved later
received_message = ""
transmitted_message = "Application is running"

# Event to signal the threads to stop
stop_event = threading.Event()

@app.route('/message', methods=['POST'])
def receive_message():
    global received_message
    data = request.json
    received_message = data.get('message', '')
    print(f"Received message: {received_message}")
    return jsonify({'status': 'Message received', 'message': received_message}), 200

@app.route('/message', methods=['GET'])
def send_message():
    global transmitted_message
    print(f"Sending message: {transmitted_message}")
    return jsonify({'message': transmitted_message}), 200

def run_flask():
    app.run(host='0.0.0.0', port=5000)

import torch
import jetson_inference
import jetson_utils
import time
import traitlets
from jetbot import Robot, Camera
device = torch.device('cuda')

import torch
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('best_steering_model_xy_trt.pth')) # well trained road following model

model_trt_collision = TRTModule()
model_trt_collision.load_state_dict(torch.load('best_model_trt.pth')) # well trained collision avoidance model

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

from IPython.display import display
import ipywidgets
import traitlets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=224, height=224, fps=30)

from jetbot import Robot

robot = Robot()

import math

angle = 0.0
angle_last = 0.0
count_stops = 0
go_on = 1
stop_time = 10 # The number of frames to remain stopped
x = 0.0
y = 0.0
speed_value = 0.09
threshold=0.95
prob_blocked=0
def execute(change):
    global angle, angle_last, blocked_slider, robot, count_stops, stop_time, go_on, x, y, blocked_threshold
    global speed_value, steer_gain, steer_dgain, steer_bias
    global prob_blocked          
    steer_gain = 0.01
    steer_dgain = 0.003
    steer_bias = 0
       
    image_preproc = preprocess(change['new']).to(device)
     
    #Collision Avoidance model:
    
    prob_blocked = float(F.softmax(model_trt_collision(image_preproc), dim=1).flatten()[0])
    
    stop_time=10
    if go_on == 1:    
        if prob_blocked > threshold: # threshold should be above 0.5
            count_stops += 1
            go_on = 2
        else:
            #start of road following detection
            go_on = 1
            count_stops = 0
            xy = model_trt(image_preproc).detach().float().cpu().numpy().flatten()        
            x = xy[0]            
            y = (0.5 - xy[1]) / 2.0
            speed_value = 0.09
    else:
        count_stops += 1
        if count_stops < stop_time:
            x = 0.0 #set x steering to zero
            y = 0.0 #set y steering to zero
            speed_value = 0 # set speed to zero (can set to turn as well)
            robot.left_motor.value = -0.30  # Negative value to turn left motor backward
            robot.right_motor.value = 0.30
        else:
            go_on = 1
            count_stops = 0
            
    
    angle = math.atan2(x, y)        
    pid = angle * steer_gain + (angle - angle_last) * steer_dgain
    steer_val = pid + steer_bias 
    angle_last = angle
    if go_on==1:
        robot.left_motor.value = max(min(speed_value + steer_val, 1.0), 0.0)
        robot.right_motor.value = max(min(speed_value - steer_val, 1.0), 0.0) 


# Load the pre-trained TensorRT engine model and labels
model_path = "models/dataset/ssd-mobilenet.onnx"
labels_path = "models/dataset/labels.txt"

net = jetson_inference.detectNet(argv=["--model=" + model_path, "--labels=" + labels_path, "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes","--threshold=0.5"])


# Define object of interest (label index as per your labels.txt)
  # Change this to the name of the object you want to follow
OBJECT_LABEL=""
# Function to find the desired object
def find_object(detections):
    global OBJECT_LABEL
    for detection in detections:
        if net.GetClassDesc(detection.ClassID) == OBJECT_LABEL:
            return detection
    return None

# Function to navigate towards the object normal değerler 0.1 0.0015
def navigate_to_object(detection):
    center_x = detection.Center[0]
    image_width = camera.width
    error_x = center_x - image_width / 2

    # Define control gains
    linear_speed = 0.10  # Forward speed
    angular_gain = 0.0028  # Turning gain

    # Calculate turn speed based on error
    turn_speed = error_x * angular_gain

    # Move the robot
    robot.set_motors(linear_speed + turn_speed, linear_speed - turn_speed)


def emiran_hizalama(plaka):
    if plaka == "6" or plaka == "7":
        alt_sinir = 102
        ust_sinir = 110 
    else:    
        alt_sinir = 107
        ust_sinir = 115
    for i in range(8):
        img = camera.value
        cuda_img = jetson_utils.cudaFromNumpy(img)
        detections = net.Detect(cuda_img)
        for detection in detections:
            if net.GetClassDesc(detection.ClassID) == plaka:
                detected = detection
                if ust_sinir >detected.Center[0] > alt_sinir:
                    print(detected.Center[0])
                    return
                elif detected.Center[0] < alt_sinir:
                    turn_left()
                    robot.forward(0.14)
                    time.sleep(0.15)
                    robot.stop()
                    turn_right()
                    time.sleep(0.2)
                    alligning_func(tolerance=0.2)
                elif detected.Center[0] > ust_sinir:
                    turn_right()
                    robot.forward(0.14)
                    time.sleep(0.15)
                    robot.stop()
                    turn_left()
                    time.sleep(0.2)
                    alligning_func(tolerance=0.2)

# Main loop
park_completed_flag=0
def final_to_object():
    counter_for_object=0
    global received_message,transmitted_message,count,durum,park_completed_flag,threshold

    while True:
        if counter_for_object==100:
            durum=0
            threshold=0.95
            received_message=""
            count=0
            transmitted_message=f"Park has been completed, Total penalty points: {toplamceza}"
            park_completed_flag=1
            break
        else:
            img = camera.value
            cuda_img = jetson_utils.cudaFromNumpy(img)

            detections = net.Detect(cuda_img)


            object_detection = find_object(detections)
            if object_detection:
                navigate_to_object(object_detection)
            else:
                robot.stop()
                counter_for_object=counter_for_object+1
            
        time.sleep(0.01)
  
      
def obcjet_control():
    for i in range(10):
        time.sleep(0.05)
        img = camera.value
        cuda_img = jetson_utils.cudaFromNumpy(img)

        detections = net.Detect(cuda_img)

        object_detection = find_object(detections)
        if object_detection:
            if 90<object_detection.Center[0]<130: 
                print(net.GetClassDesc(object_detection.ClassID))
                return True
    return False


def control_left():
    robot.set_motors(-0.12,0.15)
    time.sleep(0.06)
    robot.stop()

def control_right():
    robot.set_motors(0.15,-0.14)
    time.sleep(0.06)
    robot.stop()

import Jetson.GPIO as GPIO

# GPIO pin setup
TRIGGER_PIN_left = 23
ECHO_PIN_left = 24
TRIGGER_PIN_right = 37
ECHO_PIN_right = 38

# GPIO setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIGGER_PIN_left, GPIO.OUT)
GPIO.setup(ECHO_PIN_left, GPIO.IN)
GPIO.setup(TRIGGER_PIN_right, GPIO.OUT)
GPIO.setup(ECHO_PIN_right, GPIO.IN)


GPIO.setup(11,GPIO.IN)
GPIO.setup(13,GPIO.IN)
toplamceza = 0
def kirmizi():
    global toplamceza
    global transmitted_message
    durum_kirmizi = 0
    while True:        
        if durum_kirmizi == 0:
            if GPIO.input(11) or GPIO.input(13):
                transmitted_message = f"Crossed over the red line: {toplamceza}"
                toplamceza += 1
                durum_kirmizi = 1
        else:
            if GPIO.input(11) == 0 and GPIO.input(13) == 0:
                durum_kirmizi = 0      
        time.sleep(0.0002)


def get_distance_left(timeout=0.3):
    global TRIGGER_PIN_left,ECHO_PIN_left
    while True:
        timeoutx=1
        timeouty=1
        # Set trigger to HIGH
        time.sleep(0.05)

        GPIO.output(TRIGGER_PIN_left, GPIO.HIGH)

        # Set trigger after 0.01ms to LOW
        time.sleep(0.001)
        GPIO.output(TRIGGER_PIN_left, GPIO.LOW)

        # Save start time
        start_time = time.time()
        stop_time = time.time()

        # Wait for the echo pin to go high
        timeout_start = time.time()
        while GPIO.input(ECHO_PIN_left) == 0 and timeoutx:
            start_time = time.time()
            if start_time - timeout_start > timeout:
                timeoutx=0
                break  # Restart the function if a timeout occurs
        if timeoutx==0:
            continue
        # Wait for the echo pin to go low
        timeout_start = time.time()
        while GPIO.input(ECHO_PIN_left) == 1 and timeouty:
            stop_time = time.time()
            if stop_time - timeout_start > timeout:
                timeouty=0
                break  # Restart the function if a timeout occurs
        if timeouty==0:
            continue
        # Time difference between start and arrival
        time_elapsed = stop_time - start_time

        # Multiply with the sonic speed (34300 cm/s) and divide by 2, because there and back
        distance = (time_elapsed * 34300) / 2
        return distance

def get_distance_right(timeout=0.3):
    global ECHO_PIN_right,TRIGGER_PIN_right
    while True:
        timeoutx=1
        timeouty=1
        # Set trigger to HIGH
        time.sleep(0.05)
        GPIO.output(TRIGGER_PIN_right, GPIO.HIGH)

        # Set trigger after 0.01ms to LOW
        time.sleep(0.001)
        GPIO.output(TRIGGER_PIN_right, GPIO.LOW)

        # Save start time
        start_time = time.time()
        stop_time = time.time()

        # Wait for the echo pin to go high
        timeout_start = time.time()
        while GPIO.input(ECHO_PIN_right) == 0 and timeoutx:
            start_time = time.time()
            if start_time - timeout_start > timeout:
                timeoutx=0
                break
                  # Restart the function if a timeout occurs
        if timeoutx==0:
            continue
        # Wait for the echo pin to go low
        timeout_start = time.time()
        while GPIO.input(ECHO_PIN_right) == 1 and timeouty:
            stop_time = time.time()
            if stop_time - timeout_start > timeout:
                timeouty=0
                break
        if timeouty==0:
            continue
        # Time difference between start and arrival
        time_elapsed = stop_time - start_time

        # Multiply with the sonic speed (34300 cm/s) and divide by 2, because there and back
        distance = (time_elapsed * 34300) / 2
        return distance


def alligning_func(tolerance=0.5):
    while True:
        time.sleep(0.05)
        # Calculate the difference between the average distances

        right=get_distance_right()

        time.sleep(0.2)

        left=get_distance_left()

        distance_diff = left-right

        # Check alignment
        if abs(distance_diff) <= tolerance:
            robot.stop()
            return
        elif distance_diff > 0:
            # If left is greater, turn right
            control_right()
        else:
            # If right is greater, turn left
            control_left()

          # Delay between adjustments
def turn_left():
    robot.left(0.15)
    time.sleep(0.8)
    robot.stop() 
def turn_right():    
    robot.right(0.15)
    time.sleep(0.66)
    robot.stop()
durum=0
count=0

# Create and start the Flask thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

Red_line = threading.Thread(target=kirmizi)
Red_line.start()


toplamceza=0
time_out_counter=0


try:
    while True:
        if durum==0:
            if received_message in ["1","2","3","4","5","6","7","8","9","10"]:
                OBJECT_LABEL=received_message
                durum=1
                transmitted_message="Park is started"
                time.sleep(0.3)
                park_completed_flag=0
                toplamceza = 0
                time_out_counter=time.time()
        elif durum in [1,2,3]:
            if time.time()-time_out_counter>170:
                time_out_counter=0
                robot.stop()
                toplamceza=0
                durum=0
                threshold=0.95
                received_message=""
                count=0
                park_completed_flag=0
                transmitted_message="Plate is not found"
                continue
            execute({'new': camera.value})
            time.sleep(0.05)
            if durum==1: 
                count=count+1
                if count==200:
                    durum=2
                    threshold=1
            elif durum==2:
                if prob_blocked>0.93:
                    turn_right() 
                    turn_right()
                    durum=3
                    time.sleep(0.3)
            elif durum==3:
                if prob_blocked>0.94:
                    turn_right() 
                    turn_right()
                    time.sleep(1)
                else:
                    if (get_distance_left()//1 in [96,97,78,79,62,61,45,46,28,27]):
                        robot.stop()
                        time.sleep(0.2)
                        turn_right()
                        alligning_func()

                        robot.forward(0.17)
                        time.sleep(0.7)
                        robot.stop()

                        alligning_func()

                        if(obcjet_control()==True):
                            emiran_hizalama(OBJECT_LABEL)
                            final_to_object()
                        if park_completed_flag==1:
                            continue

                        robot.backward(0.17)
                        time.sleep(0.7)
                        robot.stop()

                        turn_right()

                        time.sleep(0.4)

                        turn_right()

                        alligning_func()

                        robot.forward(0.17)
                        time.sleep(0.7)
                        robot.stop()

                        alligning_func()


                        if(obcjet_control()==True):
                            emiran_hizalama(OBJECT_LABEL)
                            final_to_object()
                        #time.sleep(2)
                        if park_completed_flag==1:
                            continue
                        robot.backward(0.17)
                        time.sleep(0.7)
                        robot.stop()

                        turn_right()
                        alligning_func() 
                        robot.forward(0.15)
                        time.sleep(0.5)
                        robot.stop() 
except KeyboardInterrupt:
    robot.stop()
    toplamceza=0
    durum=0
    threshold=0.95
    received_message=""
    count=0
    park_completed_flag=0
    transmitted_message="Function is interrupted"
    time_out_counter=0




