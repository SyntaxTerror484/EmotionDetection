from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
from keras.models import load_model
import numpy as np
from keras.utils import img_to_array
from keras.preprocessing import image
from PIL import ImageGrab, Image
import streamlit as st
import sys
import time
from io import StringIO

emotion_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
colors = {0:(0, 128, 255), 1:(246, 61, 252), 2:(246, 61, 252), 3:(0,255,90), 4:(204, 204, 204), 5:(237, 28, 63), 6:(252,238,33)}

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

prediction_model = load_model('model')
age_model = load_model('amongus')


def cam():


    class VideoProcessor:
        def recv(self, frame):
            frame_ = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 3)

            for x, y, w, h in faces:
                cv2.rectangle(frame_, (x, y), (x + w, y + h), (0,255,0), 3)

                roi = gray[y : y + h, x : x + w]
                roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi]) != 0:
                    roi_ = roi.astype('float')/255.0
                    roi_ = img_to_array(roi_)
                    roi_ = np.expand_dims(roi_, axis=0)

                    prediction = prediction_model.predict(roi_)[0]
                    emotion = emotion_labels[prediction.argmax()]
                    age = str(int(age_model.predict(roi_)))
                    label_pos = (x, y - 10)

                    cv2.putText(frame_, f'{emotion}, {age}', label_pos, cv2.FONT_HERSHEY_PLAIN, 2, colors[prediction.argmax()], 2)

            return av.VideoFrame.from_ndarray(frame_, format='bgr24')

    webrtc_streamer(key="daf", video_processor_factory=VideoProcessor)


def upload():

    global emotion_labels
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)


        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 3)
        k = np.asarray(image)


        for x, y, w, h in faces:


            roi = gray[y: y + h, x: x + w]
            roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi]) != 0:
                roi_ = roi.astype('float') / 255.0
                roi_ = img_to_array(roi_)
                roi_ = np.expand_dims(roi_, axis=0)

                prediction = prediction_model.predict(roi_)[0]
                emotion = emotion_labels[prediction.argmax()]
                age = str(int(age_model.predict(roi_)))
                label_pos = (x, y - 10)
                cv2.rectangle(k, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(k, f'{emotion}, {age}', label_pos, cv2.FONT_HERSHEY_PLAIN, 2, colors[prediction.argmax()],
                            2)
        data = Image.fromarray(k)


            # st.write(emotion)
            # st.write(age)

        st.image(data)
# st.image(image)


st.sidebar.title('Choose:')
k = st.sidebar.radio("", ('Camera','Upload'))
if k == 'Camera':
    cam()
if k == 'Upload':
    upload()
