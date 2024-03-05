from ultralytics import YOLO
from sort import Sort
import numpy as np
import math
import cv2

# Tracker and will be utilized for object tracking.
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# We will Use YOLOv8 for object detection
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("road.mp4")

# Yolo will be applied only on a sub region of the frame for optimization.
mask  = cv2.imread("road_mask.png")
# cap.set(3, 1080)
# cap.set(4, 720)

# This is the line after which we increse the counter of the counted vehicle.
limits = [730, 560, 1250, 560]
count = 0
idSet = set()

frame_count = 0
while True:
    if frame_count%15!=0:   # Applying 
        break
    _, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    result = model(imgRegion, stream=True)
    detections = np.empty((0,5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2= int(x1), int(y1), int(x2), int(y2)
            cls_idx = int(box.cls)
            cls = model.model.names[cls_idx]   # model.model.names is a dictionary of trained classes
            conf = math.ceil((box.conf[0]*100)) / 100
            
            if cls=="car" or cls=="truck" and conf>0.3:
                #cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
                #cv2.putText(img, cls, (x1,y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))
    
    resultsTracker = tracker.update(detections)
    
    
    for result in resultsTracker:
        Id = 0
        x1,y1,x2,y2,Id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (0,0,255),cv2.FILLED)
        counted = False
        
            
        if limits[0]<cx<limits[2] and limits[1]-10<cy<limits[1]+10:
            if Id not in idSet:
                count+=1
                idSet.add(Id)
        cv2.putText(img, str(count), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),2)
        #Id = int(Id)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 2)
        cv2.putText(img, str(int(Id)), (x1+5,y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),2)
        
        #print(Id)
    
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 2)
    
    cv2.imshow("result", img)
    #cv2.imshow("ROI", imgRegion)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()