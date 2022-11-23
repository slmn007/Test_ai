from flask import Flask, render_template, Response

import numpy as np
import cv2

import face_recognition
from tensorflow.keras.models import load_model

app = Flask(__name__)
camera = cv2.VideoCapture(0)

model_path = "model/faceClassification3.h5"
model_train = load_model(model_path)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            face_locations = face_recognition.face_locations(frame)
            for i in range(len(face_locations)):
                x1=face_locations[i][3]
                y1=face_locations[i][0]         
                x2=face_locations[i][1]            
                y2=face_locations[i][2]    
                
                tempImg=frame[y1:y2+1,x1:x2+1]
                #tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2BGR)
                inp = cv2.resize(tempImg, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                inp=np.expand_dims(inp,axis=0)
                faceClass = model_train.predict(inp)
                temp=np.argmax(faceClass,axis=1)
                clr=((0,255,0),(0,255,255),(0,0,255))
                label=['Masker Penuh','Masker Tidak Penuh','Tanpa Masker']
                
                #Hijau-Kelas 0 : Menggunakan Full Masker
                #Kuning-kelas 1 : Menggunakan setengah Masker
                #Merah-Kelas 2 : Tanpa Makser
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), clr[int(temp)], 2)
                (w, h), _ = cv2.getTextSize(label[int(temp)], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
                frame = cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), clr[int(temp)], -1)
                frame = cv2.putText(frame, label[int(temp)], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

            
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    
    app.run()