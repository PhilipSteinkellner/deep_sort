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
#from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque

ap = argparse.ArgumentParser(description="Run Tracking on an input video")
ap.add_argument("-i", "--input", help="path to input video", default = "./input.mp4")
ap.add_argument("-c", "--class", nargs='+', help="names of classes to track", default = "person")
args = vars(ap.parse_args())
video_input = args["input"]
classes_to_track = [cl.lower() for cl in args["class"]]

pts = [deque(maxlen=30) for _ in range(100)]
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
inpWidth = 416 # Width of network's input image
inpHeight = 416 # Height of network's input image

# class names
classesFile = "../yolo-coco/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# TRACKER parameters
distance_metric = "cosine"
max_cosine_distance = 0.2
max_euclidean_distance = 0.2 # for distance metric
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric(distance_metric, max_cosine_distance, nn_budget)
max_iou_distance=0.9
max_age=30
n_init=3

def main(yolo):

    start = time.time() # overall computation time

    # checkpoint file for deep assossiation matrix
    model_filename = 'resources/networks/mars-small128.pb'
    # feature extractor
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    video_capture = cv2.VideoCapture(video_input)
    # video features
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # init tracker
    tracker = Tracker(metric, [video_width, video_height], max_euclidean_distance, max_iou_distance, max_age, n_init)

    writeVideo_flag = True
    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        output_name = video_input.split(".")[0] + "_output_{}.mp4".format(datetime.now().strftime("%m%d-%H%M"))
        video_writer = cv2.VideoWriter('./output/'+output_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (video_width, video_height))
        tracks_file = open(video_input.split(".")[0]+".txt", 'w')

    win_name = "YOLO_Deep_SORT_TRACKER"
    #cv2.namedWindow(win_name, flags=cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(win_name, video_width, video_height)
    
    frame_index = 1
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
        features = encoder(frame, boxs) # get appearance feature vector

        detections = [Detection(bbox, conf, feature, cl) for bbox, conf, feature, cl in 
                      zip(boxs, confidences, features, class_names) if cl in classes_to_track]

        # Call the tracker
        t2 = time.time()
        tracker.predict()
        tracker.update(detections)
        track_time = time.time() - t2
        
        active_tracks = 0
        indexIDs = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 3) # frame, top, left, bottom, right, colour, thickness

        for track in tracker.tracks:
            if not track.is_confirmed():# or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[active_tracks] % len(COLORS)]]
            color = [0,255,0] # green bgr

            # bbox
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            # track ID
            cv2.putText(frame, "track_id: "+str(track.track_id),(int(bbox[0]), int(bbox[1] -25)),0, 0.75, (color),2)
            # track class
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
            
            bbox = track.to_tlwh()
            # save tracks to file
            tracks_file.write(str(frame_index)+' ')
            tracks_file.write("{} {} {} {} {} {}".format(track.get_id(), bbox[0], bbox[1], bbox[2], bbox[3], track.get_conf()))
            tracks_file.write(' ' + '-1, -1, -1')
            tracks_file.write('\n')

        count = len(set(counter))
        # top left on screen
        cv2.putText(frame, "Total Object Counter: "+str(count),(10, 50),0, 1, (0,0,0),11)
        cv2.putText(frame, "Total Object Counter: "+str(count),(10, 50),0, 1, (255,255,255),1)
        cv2.putText(frame, "Current Object Counter: "+str(active_tracks),(10, 100),0, 1, (0,0,0),11)
        cv2.putText(frame, "Current Object Counter: "+str(active_tracks),(10, 100),0, 1, (255,255,255),1)
        cv2.putText(frame, "FPS: %.2f, YOLO: %.2fms, SORT: %.2fms"%(fps, det_time*1000, track_time*1000),(10, 150),0, 1, (0,0,0),11)
        cv2.putText(frame, "FPS: %.2f, YOLO: %.2fms, SORT: %.2fms"%(fps, det_time*1000, track_time*1000),(10, 150),0, 1, (255,255,255),1)
        cv2.putText(frame, "maxCosineDist: %.2f, maxEuclideanDist: %.2f"%(max_cosine_distance, max_euclidean_distance),(10, 200),0, 1, (0,0,0),11)
        cv2.putText(frame, "maxCosineDist: %.2f, maxEuclideanDist: %.2f"%(max_cosine_distance, max_euclidean_distance),(10, 200),0, 1, (255,255,255),1)

        # bottom left on screen
        cv2.putText(frame, "Frame: %s/%s"%(frame_index, total_frame_count),(10, video_height-20),0, 1, (0,0,0),11)
        cv2.putText(frame, "Frame: %s/%s"%(frame_index, total_frame_count),(10, video_height-20),0, 1, (255,255,255),1)
        cv2.putText(frame, "confTresh: %.2f, nmsTresh: %.2f"%(confThreshold, nmsThreshold),(10, video_height-70),0, 1, (0,0,0),11)
        cv2.putText(frame, "confTresh: %.2f, nmsTresh: %.2f"%(confThreshold, nmsThreshold),(10, video_height-70),0, 1, (255,255,255),1)
        cv2.putText(frame, "%s"%(modelWeights.rsplit(".")[-2] + " {}x{}".format(inpWidth, inpHeight)),(10, video_height-120),0, 1, (0,0,0),11)
        cv2.putText(frame, "%s"%(modelWeights.rsplit(".")[-2] + " {}x{}".format(inpWidth, inpHeight)),(10, video_height-120),0, 1, (255,255,255),1)
        cv2.imshow(win_name, frame)
        #cv2.imwrite('/tmp/%08d.jpg'%frame_index,frame)

        if writeVideo_flag:
            #save a frame
            video_writer.write(frame)
                        
        fps  = 1./(time.time()-t1)
        print("Progress: %.2f%% (%d/%d) || FPS: %.2f || YOLO: %.2fms || Deep_Sort: %.2fms"
            %(100*(frame_index)/total_frame_count, frame_index, total_frame_count, fps, det_time*1000, track_time*1000))
        
        frame_index = frame_index + 1

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    print("[Finished in {} seconds]".format(round(end-start)))
    print("[Wrote outpute file to ./output/{}]".format(output_name))

    video_capture.release()
    if writeVideo_flag:
        video_writer.release()
        tracks_file.close()
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