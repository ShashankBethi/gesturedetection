import cv2
import numpy as np
import math

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve accuracy
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in the edged frame
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the convex hull of the hand contour
        hull = cv2.convexHull(hand_contour)
        
        # Draw the hull on the frame
        cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
        
        # Calculate the centroid of the hand
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Calculate distance from centroid to farthest point on convex hull
            max_dist = 0
            farthest_point = None
            for point in hand_contour:
                dist = math.dist((cx, cy), tuple(point[0]))
                if dist > max_dist:
                    max_dist = dist
                    farthest_point = tuple(point[0])
            
            # Draw a circle around the farthest point
            cv2.circle(frame, farthest_point, 5, (255, 0, 0), -1)
            
            # Determine gesture based on distance
            if max_dist > 100:
                cv2.putText(frame, "Gesture Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
