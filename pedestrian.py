import cv2
import numpy as np

pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_pedestrians(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    return frame, len(pedestrians)

def traffic_signal_on_frame(pedestrian_count, frame, road_name):
    if pedestrian_count > 0:
        signal = "RED - STOP (Pedestrians Crossing)"
        color = (0, 0, 255) 
    else:
        signal = "GREEN - GO (No Pedestrians)"
        color = (0, 255, 0) 

    cv2.putText(frame, f"Road: {road_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Pedestrians: {pedestrian_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Signal: {signal}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

def resize_frame(frame, size=(640, 480)):
    return cv2.resize(frame, size)

videos = ['road5.mp4', 'road6.mp4', 'road7.mp4', 'road8.mp4']

cap1 = cv2.VideoCapture(videos[0])
cap2 = cv2.VideoCapture(videos[1])
cap3 = cv2.VideoCapture(videos[2])
cap4 = cv2.VideoCapture(videos[3])
 
common_size = (640, 480)
 
while True: 
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not ret1 or not ret2 or not ret3 or not ret4:
        print("One or more videos have ended.")
        break
 
    frame1 = resize_frame(frame1, common_size)
    frame2 = resize_frame(frame2, common_size)
    frame3 = resize_frame(frame3, common_size)
    frame4 = resize_frame(frame4, common_size)
 
    frame1, count1 = detect_pedestrians(frame1)
    frame2, count2 = detect_pedestrians(frame2)
    frame3, count3 = detect_pedestrians(frame3)
    frame4, count4 = detect_pedestrians(frame4)
 
    frame1 = traffic_signal_on_frame(count1, frame1, "Road 1")
    frame2 = traffic_signal_on_frame(count2, frame2, "Road 2")
    frame3 = traffic_signal_on_frame(count3, frame3, "Road 3")
    frame4 = traffic_signal_on_frame(count4, frame4, "Road 4")
 
    combined_frame_top = np.hstack((frame1, frame2)) 
    combined_frame_bottom = np.hstack((frame3, frame4)) 
    combined_frame = np.vstack((combined_frame_top, combined_frame_bottom))   

    cv2.namedWindow('Pedestrian Detection and Traffic Signal Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pedestrian Detection and Traffic Signal Control', combined_frame.shape[1], combined_frame.shape[0])

    cv2.imshow('Pedestrian Detection and Traffic Signal Control', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
