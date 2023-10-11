import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the lower and upper HSV threshold values for green objects
lower_green = np.array([50*.5, .38*255, .30*255])
upper_green = np.array([90*.5,   1*255, .80*255])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask using the HSV threshold values
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # For each contour, draw a bounding box around the object
    x, y, w, h = cv2.boundingRect(mask)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
