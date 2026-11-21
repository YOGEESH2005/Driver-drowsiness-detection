import os
import time
from collections import deque

import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer


mixer.init()
sound = mixer.Sound(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alarm.wav'))
sound.set_volume(1.0)

base_dir = os.path.dirname(os.path.abspath(__file__))
face = cv2.CascadeClassifier(os.path.join(base_dir, 'haarcascade', 'haarcascade_frontalface_alt.xml'))
leye = cv2.CascadeClassifier(os.path.join(base_dir, 'haarcascade', 'haarcascade_lefteye_2splits.xml'))
reye = cv2.CascadeClassifier(os.path.join(base_dir, 'haarcascade', 'haarcascade_righteye_2splits.xml'))
eyes = cv2.CascadeClassifier(os.path.join(base_dir, 'haarcascade', 'haarcascade_eye.xml'))

lbl=['Closed eyes','Open eyes']

model = load_model(os.path.join(base_dir, 'CNN__model.h5'))
path = base_dir
camera_index = 0
cap = None
for idx in range(5):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        camera_index = idx
        break
    cap.release()

if cap is None or not cap.isOpened():
    raise SystemExit('No webcam found. Connect a camera and try again.')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
window_name = 'Driver drowsiness detection'
CLOSURE_WINDOW_SECONDS = 10
CLOSED_EYE_RATIO_THRESHOLD = 0.70
MIN_EYE_OBSERVATIONS = 20
closure_samples = deque()
alarm_active = False
alarm_latched = False
manual_silence = False
saved_alarm_image = False


def maybe_save_alarm_frame(frame):
    global saved_alarm_image
    if not saved_alarm_image:
        cv2.imwrite(os.path.join(base_dir, 'image.jpg'), frame)
        saved_alarm_image = True


cv2.namedWindow(window_name)

while(True):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f'Unable to read from webcam index {camera_index}.')
        break
    height,width = frame.shape[:2]
    rpred = [99]
    lpred = [99]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) #Face detection
    eye=eyes.detectMultiScale(gray) #eye detection
    left_eye = leye.detectMultiScale(gray) #Left eye detection
    right_eye =  reye.detectMultiScale(gray) #Right eye detection
    
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) ,thickness=cv2.FILLED)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (150,150,150) , 1) 
        
    for (x,y,w,h) in eye:
        cv2.rectangle(frame, (x,y),(x+w,y+h) ,(150,150,150) , 1) 
       
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(100,100)) 
        r_eye= r_eye/255                  
        r_eye=  r_eye.reshape(100,100,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = [int(model.predict(r_eye, verbose=0)[0][0] >= 0.5)]

        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(100,100))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(100,100,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = [int(model.predict(l_eye, verbose=0)[0][0] >= 0.5)]
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    detected_eye_states = []
    if rpred[0] != 99:
        detected_eye_states.append(rpred[0] == 0)
    if lpred[0] != 99:
        detected_eye_states.append(lpred[0] == 0)

    sample_time = time.monotonic()
    closed_eye_count = sum(detected_eye_states)
    closure_samples.append((sample_time, closed_eye_count, len(detected_eye_states)))
    window_start = sample_time - CLOSURE_WINDOW_SECONDS
    while closure_samples and closure_samples[0][0] < window_start:
        closure_samples.popleft()

    window_complete = bool(closure_samples) and sample_time - closure_samples[0][0] >= CLOSURE_WINDOW_SECONDS
    eye_observations = sum(count for _, _, count in closure_samples)
    closed_eye_observations = sum(closed for _, closed, _ in closure_samples)
    closed_ratio = (
        closed_eye_observations / eye_observations if eye_observations else 0.0
    )
    eye_detection_coverage = (
        sum(count > 0 for _, _, count in closure_samples) / len(closure_samples)
        if closure_samples
        else 0.0
    )
    alarm_condition = (
        window_complete
        and eye_observations >= MIN_EYE_OBSERVATIONS
        and closed_ratio > CLOSED_EYE_RATIO_THRESHOLD
    )

    if manual_silence:
        if not alarm_condition:
            manual_silence = False
            alarm_latched = False
        else:
            try:
                sound.stop()
            except Exception:
                pass
            cv2.putText(frame,'Alarm muted - waiting for alert condition to clear', (10,height-20), font, 0.65,(255,255,255),1,cv2.LINE_AA)
            cv2.imshow(window_name,frame)
            cv2.waitKey(1)
            continue

    if alarm_condition and not manual_silence and not alarm_latched:
        alarm_latched = True
        maybe_save_alarm_frame(frame)

    alarm_active = alarm_latched
    if alarm_active:
        if not manual_silence and not mixer.get_busy():
            sound.play(-1)
        pulse_width = int(4 + 4 * abs(np.sin(time.monotonic() * 8)))
        cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), pulse_width)
        cv2.putText(frame,'DROWSY ALERT', (10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Press s to stop alarm', (220,height-20), font, 0.8,(255,255,255),1,cv2.LINE_AA)
    else:
        alarm_active = False
        saved_alarm_image = False
        try:
            sound.stop()
        except Exception:
            pass
        cv2.putText(frame, f'Closed: {closed_ratio:.0%}', (10,height-32), font, 0.8,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame, f'Eyes found: {eye_detection_coverage:.0%}', (10,height-8), font, 0.65,(255,255,255),1,cv2.LINE_AA)

    cv2.imshow(window_name,frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('s'), ord('S')):
        manual_silence = True
        alarm_latched = False
        sound.stop()
        alarm_active = False
cap.release()
cv2.destroyAllWindows()