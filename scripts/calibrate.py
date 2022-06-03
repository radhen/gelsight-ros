# OPTIONAL: tool for interactively selecting keypoints for images
# mouse click for selecting keypoints in the order of [topleft, topright, bottomleft, bottomright]
# when finished, press any key to exit and save the keypoints

import cv2
import scipy.io
import numpy as np
import csv 

# field names 
fields = ['Name', 'x', 'y'] 

# data rows of csv file 
coords = [ ['topleft', '0', '0'], 
        ['topright', '0', '0'], 
        ['bottomleft', '0', '0'], 
        ['bottomright', '0', '0']]

X = []
Y = []

def save_csv():
    filename = "config.csv"

    for i in range(4):
        coords[i][1] = X[i]
        coords[i][2] = Y[i]

    with open(filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(coords)

# mouse callback, store the keypoints when mouse clicked
def click_keypoints(event, x, y, flags, param):
    # grab references to the global variables
    global X, Y

    # mouse clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        X.append(x) # index + 1 for matlab consistency
        Y.append(y)
        print(x, y)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click_keypoints)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if frame == "" or frame is None:
        print("waiting for streaming")
        continue

    for i in range(len(X)):
        cv2.circle(frame, (X[i], Y[i]), 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)

    if len(X) == 4:
        break

    if c == ord('q') & 0xff:
        break
    elif c == ord('r') & 0xff:
        X = []
        Y = []

print(frame.shape)
if len(X) == 4:
    save_csv()
    print(coords)
