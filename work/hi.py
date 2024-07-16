from flask import Flask,render_template,Response
import cv2
from deepface import DeepFace
import threading
import os
import gspread
import datetime
from google.oauth2.service_account import Credentials

app = Flask(__name__)
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("file.json", scopes=scope)
client = gspread.authorize(creds)
spreadsheet_id = "1qdj6wcrxVxVsaD7ZnHpRS6x-DdI3ZLVjoYhxm3FlV_c"
sheet = client.open_by_key(spreadsheet_id).sheet1

face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

reference_dir=r"C:\Users\Lenovo\Desktop\photos"

reference_lock = threading.Lock()
reference_images={}

def load_image():
    with reference_lock:
        for filename in os.listdir(reference_dir):
            image_path=os.path.join(reference_dir,filename)
            img = cv2.imread(image_path)
            reference_images[filename]=img

load_ref_thread=threading.Thread(target=load_image)
load_ref_thread.start()
attendance={}
def update(name):
    if name not in attendance:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d %H:&M:%S")
        attendance[name]=timestamp
        print(f"attendance marked for {name} at {attendance[name]}")
        row = [name,timestamp]
        sheet.append_row(row)

def checkface(frame):
    try:
        height,width=frame.shape[:2]
        rezise_factor=0.5
        frame_resized=cv2.resize(frame,(int(width*rezise_factor),int(height*rezise_factor)))
        faces = face_cascade.detectMultiScale(frame_resized, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x,y,w,h) in faces:
            face_img=frame_resized[y:y+h,x:x+w]
            matched_name = "unknown"
            with reference_lock:
                for ref_name,ref_img in reference_images.items():
                    result = DeepFace.verify(ref_img,face_img)
                    if result['verified']:
                        matched_name = os.path.splitext(ref_name)[0]
                        break
            if matched_name == "unknown":
                cv2.putText(frame,"no match",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),5)
            else:
                cv2.putText(frame,f"match-{matched_name}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),5)
                update(matched_name)
              
    except Exception as e:
        print("error in face recognition:",e)
    return frame

class Video(threading.Thread):
    def _init_(self):
        super(Video,self)._init_()
        self.cap= cv2.VideoCapture(0)
        self.running= True
        self.frame=None
        self.frame_update_event=threading.Event()
    def run(self):
        while self.running:
            ret,frame=self.cap.read()
            if not ret:
                break
            self.frame=checkface(frame.copy())
            ret ,buffer= cv2.imencode('.jpg',self.frame,[cv2.IMWRITE_JPEG_QUALITY,75])

            self.frame=buffer.tobytes()
            self.frame_update_event.set()
            self.frame_update_event.clear()
        self.cap.release()


def myfun():
    video_thread=Video()
    video_thread.start()
   
    while True:
        video_thread.frame_update_event.wait()

     
        if not video_thread.running:
            break

        frame= video_thread.frame
        if frame is not None:
            yield(b'--frame\r\n'
                    b'content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')
        else:
            print("error: no frame exist")
        
        video_thread.frame_update_event.clear()
    video_thread.join()

@app.route('/')
def init():
    return render_template('index1.html')

@app.route('/video')
def video():
    return Response(myfun(),mimetype ='multipart/x-mixed-replace; boundary=frame')            

if __name__=="__main__":
    app.run(debug=True)