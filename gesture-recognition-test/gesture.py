import numpy as np
import cv2
import imutils
import math
import time

def nothing(x): #needed for createTrackbar to work in python.
    pass

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
#lower = np.array([2, 0, 0], dtype = "uint8")
lower = np.array([0, 50, 0], dtype = "uint8")
upper = np.array([120, 150, 255], dtype = "uint8")


cap = cv2.VideoCapture(0)

cv2.namedWindow('HSV settings')
cv2.createTrackbar('BL', 'HSV settings', 0, 255, nothing)
cv2.createTrackbar('GL', 'HSV settings', 0, 255, nothing)
cv2.createTrackbar('RL', 'HSV settings', 160, 255, nothing)
cv2.createTrackbar('BU', 'HSV settings', 255, 255, nothing)
cv2.createTrackbar('GU', 'HSV settings', 255, 255, nothing)
cv2.createTrackbar('RU', 'HSV settings', 255, 255, nothing)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (0, 0), (300, 300), (0, 0, 255), 2)

    crop_image = frame[0:300, 0:300]

    # Our operations on the frame come here
    # resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    #frame = imutils.resize(frame, width = 400)
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0) #blur
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


    BL = cv2.getTrackbarPos('BL', 'HSV settings')
    GL = cv2.getTrackbarPos('GL', 'HSV settings')
    RL = cv2.getTrackbarPos('RL', 'HSV settings')
    BU = cv2.getTrackbarPos('BU', 'HSV settings')
    GU = cv2.getTrackbarPos('GU', 'HSV settings')
    RU = cv2.getTrackbarPos('RU', 'HSV settings')

    lower = np.array([BL, GL, RL], dtype = "uint8")
    upper = np.array([BU, GU, RU], dtype = "uint8")

    skin_mask = cv2.inRange(hsv, lower, upper)

    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    #kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
	# mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    #ret, skin_mask = cv2.threshold(skin_mask, 127, 255, 0)
    #skin = cv2.bitwise_and(frame, frame, mask = skin_mask)

    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(skin_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    try:
        cnt = max(cnts, key = cv2.contourArea)
        cv2.drawContours(crop_image, [cnt], -1, (0, 255, 255), 2)

        # convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(crop_image, [hull], -1, (255, 0, 0), 2)

        # center of the contours
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(crop_image, (cX, cY), 5, (0, 0, 0), -1)

        # convexity defects
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            #cv2.circle(crop_image, far, 5, [0, 255, 0], -1)

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 5, [0, 255, 0], -1)

            '''
            # number of fingers
            if count_defects == 0:
                print('ONE / ZERO')
            elif count_defects == 1:
                print('TWO')
            elif count_defects == 2:
                print('THREE')
            elif count_defects == 3:
                print('FOUR')
            elif count_defects == 4:
                print('FIFE')
            else:
                pass
            '''
    except:
        pass

    #cv2.imshow("images", np.hstack([frame, skin]))
    #cv2.imshow("images", np.hstack([frame, skin_mask]))
    cv2.imshow('mask', skin_mask)
    cv2.imshow('camera', frame)
    #cv2.imshow('image', crop_image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
