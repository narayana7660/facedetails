import cv2
print(cv2.__version__)
from deepface import DeepFace
import os
import datetime
import csv
import threading
import gspread
from google.oauth2.service_account import Credentials

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("file.json", scopes=scope)
client = gspread.authorize(creds)
spreadsheet_id = "1qdj6wcrxVxVsaD7ZnHpRS6x-DdI3ZLVjoYhxm3FlV_c"
sheet = client.open_by_key(spreadsheet_id).sheet1 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

reference_images = {}
reference_dir = r"C:\Users\daggu\Desktop\pics"

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
csv_filename = f'attendance_{datetime.date.today()}.csv'


with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Timestamp'])


face_match = False
matched_name = None
face_thread_running = False
lock = threading.Lock()

def check_face(frame):
    global face_match, matched_name, face_thread_running
    local_match = False
    local_name = None
    for name, ref_img in reference_images.items():
        try:
            result = DeepFace.verify(frame, ref_img.copy())
            if result['verified']:
                local_match = True
                local_name = name
                break
        except Exception as e:
            print(f"Error in face verification for {name}: {e}")
    
    with lock:
        face_match = local_match
        matched_name = local_name
        face_thread_running = False

def update_attendance(name):
    if name not in attendance:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance[name] = timestamp
        print(f"Attendance marked for {name} at {attendance[name]}")
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, timestamp])
            row = [name, timestamp]
            sheet.append_row(row)

while True:
    ret, frame = cap.read()

    if ret:
        if not face_thread_running:
            with lock:
                face_thread_running = True
            face_thread = threading.Thread(target=check_face, args=(frame.copy(),))
            face_thread.start()

        with lock:
            if face_match:
                cv2.putText(frame, f"MATCH: {matched_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                update_attendance(matched_name)
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