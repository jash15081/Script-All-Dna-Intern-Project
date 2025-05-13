import cv2
import matplotlib.pyplot as plt
import numpy as np
# Open default camera (0 = primary webcam)
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)
# 480 x 640

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Webcam Feed', img)
    
    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
