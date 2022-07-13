from torchvision.models import detection
from torchvision.ops import nms
import numpy as np
import torch
import cv2
import time
import utils.sort as sort
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='Path to input video')
    parser.add_argument('--save', help='Save output video', action='store_true')
    args = parser.parse_args()

    return args

args = parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.models as models
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
    pretrained_backbone=True).to(DEVICE)
model.eval()

from random import randint as r
COLORS = [(r(0,255),r(0,255),r(0,255)) for _ in range(30) ]


if args.video:
    print("Loading video: ", args.video)
    clip = args.video
    cap = cv2.VideoCapture(clip)
else:
    print("Opening camera..")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    time.sleep(2.0)

print("[NOTE] Starting inference..")
writer = None
tracker = sort.Sort()
while cap.isOpened():
    ret, frame = cap.read()
    if ret != True:
        print("Cannot read video.")
        break
    
    # Frame image pre-processing
    orig = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame).to(DEVICE)

    # Inference by object detection algorithm
    detections = model(frame)[0]

    # Mask only label 1 ('person') and confidence >= 0.95
    mask = (detections['labels']== 1) & (detections['scores'] >= 0.95)
    boxes = detections['boxes'][mask]
    scores = detections['scores'][mask]

    # Perform non-maximum suppression
    mask = nms(boxes, scores, 0.8).cpu()
    boxes = boxes[mask].detach().cpu().numpy()  #.astype("int")
    scores = scores[mask].detach().cpu().numpy()
    scores = np.expand_dims(scores, axis=1)
    input = np.concatenate((boxes, scores), axis=1)

    # Update object tracking algorithm.
    tracker_out = tracker.update(input) if detections['boxes'].shape[0] != 0 else tracker.update()

    # Draw bounding box and label. 
    for out in tracker_out:
        (startX, startY, endX, endY, objID) = out.astype("int")

        cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[objID%30], 2)
        label = "object ID: {}".format(int(objID))
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[objID%30], 2)
            
    # Show output frame
    cv2.imshow("Output", orig)
    c = cv2.waitKey(1)
    # press 'Esc' key to exit.
    if c == 27: 
        break

    if args.save:
        if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                writer = cv2.VideoWriter("output.avi", fourcc, 20,
                    (orig.shape[1], orig.shape[0]), True)
            
        if writer is not None:
            writer.write(orig)

if writer is not None:
    writer.release()
    print("[NOTE] Output video saved." )

cap.release()
cv2.destroyAllWindows()