import os
import pickle

import mediapipe as mp
import cv2

mphands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir in os.listdir(DATADIR):
    # if os.path.isdir(os.path.join(DATA_DIR, dir)):
        for imgpath in os.listdir(os.path.join(DATA_DIR, dir)):
            print(imgpath)
            data_aux = []

            x = []
            y = []

            img = cv2.imread(os.path.join(DATA_DIR, dir, imgpath))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x.append(x)
                        y.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x))
                        dataaux.append(y - min(y))

                data.append(dataaux)
                labels.append(dir)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()