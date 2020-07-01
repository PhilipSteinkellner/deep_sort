#! /usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import division, print_function, absolute_import
#import os
#import datetime
from timeit import time
from datetime import datetime
import warnings
import cv2
import numpy as np
import argparse
#from PIL import Image
#from yolo import YOLO^
#from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#from deep_sort.detection import Detection as ddet
from collections import deque
#from keras import backend

#backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("--input", help="path to input video", default = "./input.mp4")
# ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(50)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# YOLO network files
modelConfiguration = "../yolo-coco/yolov3.cfg"
modelWeights = "../yolo-coco/yolov3.weights"

# YOLO parameters
confThreshold = 0.2  # Confidence threshold
nmsThreshold = 0.5  # Non-maximum suppression threshold
inpWidth = 416 #608 # Width of network's input image
inpHeight = 416 #608 # Height of network's input image

# class names
classesFile = "../yolo-coco/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def main(yolo):

    start = time.time()

    # TRACKER parameters
    distance_metric = "cosine"
    max_cosine_distance = 0.2
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric(distance_metric, max_cosine_distance, nn_budget)
    max_iou_distance=0.9
    max_age=30
    n_init=3
    # init tracker
    tracker = Tracker(metric, max_iou_distance, max_age, n_init)

    # checkpoint file for deep assossiation matrix
    model_filename = 'resources/networks/mars-small128.pb'
    # feature extractor
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args["input"])
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        output_name = str(args["input"]).split(".")[0] + "_output_{}.mp4".format(datetime.now().strftime("%m%d-%H%M"))
        video_writer = cv2.VideoWriter('./output/'+output_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    win_name = "YOLO_Deep_SORT_TRACKER"
    #cv2.namedWindow(win_name, flags=cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(win_name, w, h)

    fps = 0.0
    counter = []

    while True:

        ret, frame = video_capture.read()
        if ret == False:
            break
        t1 = time.time()
        # get yolo detections
        boxs, confidences, class_names = detect_image(yolo, frame)
        det_time = time.time() -t1 # time needed for yolo object detection
        features = encoder(frame, boxs)

        detections = [Detection(bbox, conf, feature, cl) for bbox, conf, feature, cl in zip(boxs, confidences, features, class_names) if cl =='person']

        # Call the tracker
        t2 = time.time()
        tracker.predict()
        tracker.update(detections)
        track_time = time.time() - t2
        
        active_tracks = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 3) # frame, top, left, bottom, right, colour, thickness

        for track in tracker.tracks:
            if not track.is_confirmed():# or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[active_tracks] % len(COLORS)]]
            color = [0,255,0] # green bgr

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            # track ID
            #cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 1, (0,0,0),16)
            cv2.putText(frame, "track_id: "+str(track.track_id),(int(bbox[0]), int(bbox[1] -25)),0, 0.75, (color),2)
            # track class
            #cv2.putText(frame, track.cl[0],(int(bbox[0]), int(bbox[1] -20)),0, 1.5, (0,0,0),16)
            cv2.putText(frame, "class: "+track.cl,(int(bbox[0]), int(bbox[1]-5)),0, 0.75, (color),2)

            active_tracks += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            #center point
            cv2.circle(frame, (center), 1, color, thickness)

	        #draw motion path
            # for j in range(1, len(pts[track.track_id])):
            #     if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
            #        continue
            #     thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            #     cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
            #     #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
                
        cv2.putText(frame, "Total Object Counter: "+str(count),(10, 50),0, 1, (0,0,0),11)
        cv2.putText(frame, "Total Object Counter: "+str(count),(10, 50),0, 1, (255,255,255),1)
        cv2.putText(frame, "Current Object Counter: "+str(active_tracks),(10, 100),0, 1, (0,0,0),11)
        cv2.putText(frame, "Current Object Counter: "+str(active_tracks),(10, 100),0, 1, (255,255,255),1)
        cv2.putText(frame, "FPS: %f"%(fps),(10, 150),0, 1, (0,0,0),11)
        cv2.putText(frame, "FPS: %f"%(fps),(10, 150),0, 1, (255,255,255),1)
        cv2.putText(frame, "confTresh: %s, maxCosineDist: %s"%(confThreshold, max_cosine_distance),(10, 200),0, 1, (0,0,0),11)
        cv2.putText(frame, "confTresh: %s, maxCosineDist: %s"%(confThreshold, max_cosine_distance),(10, 200),0, 1, (255,255,255),1)
        cv2.putText(frame, "Frame: %s/%s"%(frame_index+1, total_frame_count),(10, h-50),0, 1, (0,0,0),11)
        cv2.putText(frame, "Frame: %s/%s"%(frame_index+1, total_frame_count),(10, h-50),0, 1, (255,255,255),1)
        cv2.putText(frame, "%s"%(modelWeights.rsplit(".")[-2]),(10, h-100),0, 1, (0,0,0),11)
        cv2.putText(frame, "%s"%(modelWeights.rsplit(".")[-2]),(10, h-100),0, 1, (255,255,255),1)
        cv2.imshow(win_name, frame)
        #cv2.imwrite('/tmp/%08d.jpg'%frame_index,frame)

        if writeVideo_flag:
            #save a frame
            video_writer.write(frame)
            # save detections to detection file
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps  = 1./(time.time()-t1)
        #print(set(counter))
        print("Progress: %.2f%% (%d/%d) || FPS: %.2f || YOLO: %d ms || Deep_Sort: %.2f ms"
            %(100*(frame_index+1)/total_frame_count, frame_index+1, total_frame_count, fps, det_time*1000, track_time*1000))
        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    print("[Finished in {} seconds]".format(round(end-start)))
    print("[Wrote outpute file to ./output/{}]".format(output_name))

    '''if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")'''

    video_capture.release()
    if writeVideo_flag:
        video_writer.release()
        list_file.close()
    cv2.destroyAllWindows()

def init_YOLO():
    # Initialise YOLO
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def detect_image(yolo, frame):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    yolo.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = yolo.forward(getOutputsNames(yolo))

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])


    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    boxes = [boxes[i[0]] for i in indices]
    confidences = [confidences[i[0]] for i in indices]
    class_names = [classes[classIds[i[0]]] for i in indices]

    return boxes, confidences, class_names

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if __name__ == '__main__':
    yolo = init_YOLO()
    main(yolo)