import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from flask import Flask, jsonify,redirect,send_file,render_template,make_response
from flask import request
import json
import base64

app = Flask(__name__)

def readb64(uri):
    # encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route('/recognize', methods=['POST'])
def get():
    data=request.data
    decoded=data.decode("ISO-8859-1")
    decoded=json.loads(decoded)
    decoded=decoded["content"]

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

    sample_img = readb64(decoded)
    # sample_img = decoded

    results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f'HAND NUMBER: {hand_no+1}')
            print('-----------------------')
            
            for i in range(2):

                print(f'{mp_hands.HandLandmark(i).name}:')
                print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(i).value]}') 
            return 'True'
    return 'False'

if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 8090, debug =False)                   