# Usage: face_capture.py emotion-name number-of-images-to-capture
# argv.#1 -- emotion-name that you want these set of images to be labelled as.
# argv.#2 -- number of images with the named emotion you want to capture.

# It generates the face crops for 
# creating the dataset. It captures the 
# frame from the video-feed from your cam
# and detects the faces in it and saves
# cropped face as a png file.

import time
import sys
import os
import logging
import numpy as np
import cv2 as cv

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src

logger = logging.getLogger('emojifier.face_capture')

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

i = 0
N = int(sys.argv[2])
EMOTION = sys.argv[1]

PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'images', EMOTION)
if not os.path.exists(PATH):
    os.makedirs(PATH)

while i < N:
	#cap.isOpened()
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Operations on the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray=cv.imread(frame, cv.IMREAD_GRAYSCALE)
    # detect the faces, bounding boxes
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw the rectangle (bounding-boxes)
    for (x,y,w,h) in faces:

        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        img_path = os.path.join(PATH, '_' + str(time.time()) + '.png')
        cv.imwrite(img_path, frame[y:y+h, x:x+w, :])

        logger.info('{i} path: {path} created'.format(i=i, path=img_path))
        i += 1
    
    cv.imshow('faces', frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
