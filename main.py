import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from flask import Flask, jsonify,redirect,send_file,render_template,make_response
from flask import request
import json
import base64

app = Flask(__name__)
# First step is to initialize the Hands class an store it in a variable

def readb64(uri):
    print(1)
    encoded_data = uri.split(',')[1]
    print(2)
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    print(3)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(4)
    return img


@app.route('/recognize', methods=['POST'])
def get():
    data=request.data
    decoded=data.decode("ISO-8859-1")
    decoded=json.loads(decoded)
    decoded=decoded["content"]
    print(decoded)
    # print(request.headers)

    mp_hands = mp.solutions.hands

    # Now second step is to set the hands function which will hold the landmarks points
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

    # Last step is to set up the drawing function of hands landmarks on the image
    # mp_drawing = mp.solutions.drawing_utils


    # Reading the sample image on which we will perform the detection
    # sample_img = cv2.imread('media/sample5.jpg')
    sample_img = readb64(decoded)


    # sample_img = cv2.imread('media/hand_landmarks.png')

    # Here we are specifing the size of the figure i.e. 10 -height; 10- width.
    # plt.figure(figsize = [10, 10])

    # Here we will display the sample image as the output.
    # plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()


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