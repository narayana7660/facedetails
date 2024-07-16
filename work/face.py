import cv2
from deepface import DeepFace
import os
import datetime


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

reference_images = {}
reference_dir = r"C:\Users\daggu\Desktop\photos" 

for filename in os.listdir(reference_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(reference_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            name = os.path.splitext(filename)[0]
            reference_images[name] = img

if not reference_images:
    print("Error: No reference images found. Check the directory.")
    exit()

attendance = {}
counter = 0

def check_face(frame):
    global face_match, matched_name
    face_match = False
    matched_name = None
    for name, ref_img in reference_images.items():
        try:
            result = DeepFace.verify(frame, ref_img.copy())
            if result['verified']:
                face_match = True
                matched_name = name
                return
        except Exception as e:
            print(f"Error in face verification for {name}: {e}")

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                check_face(frame.copy())
            except Exception as e:
                print("Error in face verification:", e)
        counter += 1

        if face_match:
            cv2.putText(frame, f"MATCH: {matched_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if matched_name not in attendance:
                attendance[matched_name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Attendance marked for {matched_name} at {attendance[matched_name]}")
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nAttendance List:")
for name, time in attendance.items():
    print(f"{name}: {time}")