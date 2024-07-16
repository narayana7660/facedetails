import cv2
import os
import threading
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_images = {} 
reference_dir = r"C:\Users\daggu\Desktop\photos" 
for filename in os.listdir(reference_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(reference_dir,filename)
        img = cv2.imread(img_path)
        if img is not None:
            name=os.path.splitext(filename)[0]
            reference_images[name] = img

if not reference_images:
    print("Error: Reference image not loaded. Check the directory.")  
    exit()


def check_face(frame):
    global face_match
    try:
        print("Checking face...")
        result = DeepFace.verify(frame, reference_img.copy())
        print("DeepFace result:", result)
        if result['verified']:
            face_match = True
        else:
            face_match = False
    except Exception as e:
        print("Error in face verification:", e)
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                check_face(frame.copy())
            except Exception as e:
                print("Error starting thread:", e)
        counter += 1

        if face_match:
            print("match")
            cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            print("no match")
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

