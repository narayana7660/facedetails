from flask import Flask,render_template,Response
import cv2

app= Flask(__name__)

def myfun():
    cap = cv2.VideoCapture(0)
    while True:
        success,frame = cap.read()
        if not success:
            break
        else:
            ret,buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route("/")
def index():
    return render_template('index1.html')

@app.route('/video')
def video():
    return Response(myfun(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=="__main__":
    app.run(debug=True)