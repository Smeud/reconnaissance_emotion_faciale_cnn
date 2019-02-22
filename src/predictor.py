import os
import sys
import cv2 as cv
import time
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src
from src.model import model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def softmax(x):
    """Calcule la probabilite pour chaque classe sachant que la somme des probabilite des classe soit egale a 1"""
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inference(sess, gray_img_input):
    
    img = gray_img_input.reshape(1, 48, 48, 1).astype(float) / 255
    
    y_c = sess.run(y_conv, feed_dict={X:img, keep_prob:2.0})
    
    y_c = softmax(y_c)
    p = np.argmax(y_c, axis=1)
    score = np.max(y_c)
    logger.debug('''softmax-out: {}, 
        predicted-index: {}, 
        predicted-emoji: {}, 
        confidence: {}'''.format(y_c, p[0], index_emo[p[0]], score))
    return p[0], score
        

def from_cam(sess):
    cap = cv.VideoCapture(0)

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    font               = cv.FONT_HERSHEY_SIMPLEX
    fontScale          = 1
    fontColor          = (255,255,255)
    lineType           = 2

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Operations on the frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect the faces, bounding boxes
        faces = face_cascade.detectMultiScale(gray, 1.5, 3)

        # draw the rectangle (bounding-boxes)
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            bottomLeftCornerOfText = (x+20,y+h+20)

            face_img_gray = gray[y:y+h, x:x+w]
            face_img_gray = cv.resize(face_img_gray, (48, 48))
            s = time.time()
            p, confidence = inference(sess, face_img_gray)
            logger.critical('model inference time: {}'.format(time.time() - s))
            
            if confidence > 0.75:
            
                img2 = emoji_to_pic[index_emo[p]]
                img2 = cv.resize(img2, (w, h))

                alpha = img2[:,:,1]/255.0

                frame[y:y+h, x:x+w, 0] = frame[y:y+h, x:x+w, 0] * (1-alpha) + alpha * img2[:,:,0]
                frame[y:y+h, x:x+w, 1] = frame[y:y+h, x:x+w, 1] * (1-alpha) + alpha * img2[:,:,1]
                frame[y:y+h, x:x+w, 2] = frame[y:y+h, x:x+w, 2] * (1-alpha) + alpha * img2[:,:,2]
                
                #Display the prediction emotion label class
                if p == 0:
                    cv.putText(frame, f'Expression de JOIE: {p}', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
                
                if p == 1:
                    cv.putText(frame, f'Expression de DEGOUT: {p}', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)

                if p == 2:
                    cv.putText(frame, f'Expression de TRISTESSE: {p}', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)

                if p == 3:
                    cv.putText(frame, f'Expression de SURPRISE: {p}', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)

                if p == 4:
                    cv.putText(frame, f'Expression de PEUR: {p}', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)

                if p == 5:
                    cv.putText(frame, f'Expression de COLERE: {p}', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
            else: 
                cv.putText(frame,'Prediction en cours...', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)

        # Display the resulting frame
        cv.imshow('gray-scale', gray)
        cv.imshow('faces', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':

    logger = logging.getLogger('emojifier.predictor')
    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model_checkpoints')
    EMOJI_FILE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'emoji')
    tf.reset_default_graph()
    
    emo_index = {'happy': 0,'disgust': 1,'sad': 2, 'surprise': 3, 'fear': 4, 'angry': 5}
    index_emo = {v:k for k,v in emo_index.items()}
    
    emoji_to_pic = {
    'happy': None,'disgust': None,'sad': None, 'surprise': None, 'fear': None, 'angry': None
    }
   
    # ATTENTION: CHANGE THE '\\' A/C TO YOUR OS
    files = glob.glob(EMOJI_FILE_PATH + '\\*.png')

    logger.info('loading the emoji png files in memory ...')
    for file in tqdm.tqdm(files):
        logger.debug('file path: {}'.format(file))
        # ATTENTION: CHANGE THE '\\' A/C TO YOUR OS
        emoji_to_pic[file.split('\\')[-1].split('.')[0]] = cv.imread(file, -1)

    X = tf.placeholder(
        tf.float32, shape=[None, 48, 48, 1]
    )
    
    keep_prob = tf.placeholder(tf.float32)

    y_conv = model(X, keep_prob)
    
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        saver.restore(sess, os.path.join(CHECKPOINT_SAVE_PATH, 'model.ckpt'))
        logger.info('Opening the camera for getting the video feed ...')
        from_cam(sess)
