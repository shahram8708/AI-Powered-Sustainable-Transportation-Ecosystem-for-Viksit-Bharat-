import cv2
import numpy as np

bg_subtractor_1 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
bg_subtractor_2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
bg_subtractor_3 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
bg_subtractor_4 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

cap1 = cv2.VideoCapture('road1.mp4') 
cap2 = cv2.VideoCapture('road2.mp4') 
cap3 = cv2.VideoCapture('road3.mp4') 
cap4 = cv2.VideoCapture('road4.mp4') 
 
def calculate_density(fg_mask): 
    count = np.count_nonzero(fg_mask)
    return count
 
def get_traffic_signal(density, threshold=5000):
    if density > threshold:
        return "GREEN"
    else:
        return "RED"
 
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    
    if not (ret1 and ret2 and ret3 and ret4):
        print("One or more videos have finished.")
        break
    
    frame1 = cv2.resize(frame1, (400, 300))
    frame2 = cv2.resize(frame2, (400, 300))
    frame3 = cv2.resize(frame3, (400, 300))
    frame4 = cv2.resize(frame4, (400, 300))

    fg_mask1 = bg_subtractor_1.apply(frame1)
    fg_mask2 = bg_subtractor_2.apply(frame2)
    fg_mask3 = bg_subtractor_3.apply(frame3)
    fg_mask4 = bg_subtractor_4.apply(frame4)
    
    density1 = calculate_density(fg_mask1)
    density2 = calculate_density(fg_mask2)
    density3 = calculate_density(fg_mask3)
    density4 = calculate_density(fg_mask4)
    
    signal1 = get_traffic_signal(density1)
    signal2 = get_traffic_signal(density2)
    signal3 = get_traffic_signal(density3)
    signal4 = get_traffic_signal(density4)

    cv2.putText(frame1, f"Road 1 Signal: {signal1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if signal1 == "RED" else (0, 255, 0), 2)
    cv2.putText(frame2, f"Road 2 Signal: {signal2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if signal2 == "RED" else (0, 255, 0), 2)
    cv2.putText(frame3, f"Road 3 Signal: {signal3}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if signal3 == "RED" else (0, 255, 0), 2)
    cv2.putText(frame4, f"Road 4 Signal: {signal4}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if signal4 == "RED" else (0, 255, 0), 2)

    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow('Traffic Monitoring', combined_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
