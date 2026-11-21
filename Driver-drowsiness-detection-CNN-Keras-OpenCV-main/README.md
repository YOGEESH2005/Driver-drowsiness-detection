# Driver-drowsiness-detection-using-Deep-Learning
Deep learning, Tensorflow, OpenCV.

Run the detection.py script with tensorflow as a backend, The script continuously monitors the the user's eye state and classifies whether the person is drowsy or not.
Train_model.py is the script used to traing the CNN classifier model.

The alarm starts when at least 70% of detected eye observations are closed over a complete 10-second window, with at least 20 eye observations. The left and right eyes are counted separately. Once triggered, the alarm keeps sounding even if the eyes reopen. Press `s` to silence it; it stays muted until the rolling-window alarm condition clears, then rearms for a future alert. Press `q` to quit.

### The model was trained on a dataset with 11000 eye images, classified under labels such as open right eye, left eye and closed right eye, left eye.

# Vedio Link
https://drive.google.com/file/d/15_1B-n15fWxpHFIXsDOWWjkjlbLdQzTF/view?usp=sharing
