import flask
import cv2 as cv2
import numpy as np
import glob
import random

app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask import request

@app.route('/', methods=['GET'])
def get_tas():
	return 'hello sagar'

@app.route('/', methods=['GET']) #<string:url>
def get_task():
	url = request.args.get('url') 

	#load yolo
	net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3_testing.cfg")

	# adding coco names from file coco.names
	classes = []
	with open("yolo-coco/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	# Load image
	img = cv2.imread(url)
	img = cv2.resize(img, (900, 700))
	height, width, channels = img.shape

	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.3:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)  
					
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)        
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			print(label)
			print(x,y,w,h)

	return {'object': "object found at 5 meters", "url" : label}

app.run()
