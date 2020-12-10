import numpy as np
import cv2 as cv
import operator


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Sorry")
        break
    
    
    # -- Image Preprocessing--
    
    frame = cv.GaussianBlur(frame, (5, 5),0)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    thresh = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, 11, 2)
    
    kernel = np.ones((5,5), np.uint8)

    thresh = cv.dilate(frame, kernel, iterations=1)
        

    # -- Finding the edge points of the polygon--    
    
    contours, h = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    #cv.drawContours(img, contours[0], -1, (0,255,0), 20)

    polygon = contours[0] 
    
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    
    corners_of_sudoku = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    img_copy = frame.copy()

    for point in corners_of_sudoku:
        img_copy = cv.circle(img_copy, tuple(int(x) for x in point), 5, (0,0,255), -1)


    cv.imshow('img_copy',img_copy)
    


    cv.imshow('img', frame)
    #cv.imshow('frame', frame)
    if cv.waitKey(1)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("\n"+str(bottom_right)+" "+str(bottom_left)+" "+str(top_right)+" "+str(top_left))


for pt in polygon:
    print(pt[0][0], pt[0][1])

