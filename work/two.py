import cv2
from deepface import DeepFace
import os
import datetime
import csv
import pandas as pd


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
frame_count = 0
attendance_threshold = 20  
matched_name = None  


csv_file = "face details.csv"
csv_columns = ['Name', 'Time']

def write_attendance_to_csv(name, time):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writerow({'Name': name, 'Time': time})

def check_face(frame):
    global matched_name, frame_count
    for name, ref_img in reference_images.items():
        try:
            result = DeepFace.verify(frame, ref_img.copy())
            if result['verified']:
                if name not in attendance:
                    attendance[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    write_attendance_to_csv(name, attendance[name])  
                    print(f"Attendance marked for {name} at {attendance[name]}")
                frame_count = 0  
                matched_name = name  
                return True
        except Exception as e:
            print(f"Error in face verification for {name}: {e}")
    return False


while True:
    ret, frame = cap.read()

    if ret:
        frame_count += 1
        if frame_count % attendance_threshold == 0:
            try:
                if check_face(frame.copy()):
                    cv2.putText(frame, f"MATCH: {matched_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO MATCH", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                print("Error in face verification:", e)

        cv2.imshow("video", frame)



    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


def export_csv_to_excel(csv_file, excel_file):
    df = pd.read_csv(csv_file)
    df.to_excel(excel_file, index=False)
    print(f"Attendance data exported to {excel_file}")

export_csv_to_excel(csv_file, "attendance.xlsx")


print("\nAttendance List:")
for name, time in attendance.items():
    print(f"{name}: {time}")