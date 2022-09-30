################### My approach #####################
# we first start capturing video. Simple code and we exit the window using the Esc key.
# Second start object tracking using CAMshift first and provide the haarcascade of frontal face features.
# Determine the area where blur is to be applied. store it in variable and provide that variable to blur function
# Next the area determined for blur is given the result of blur for application
# Display the frame


# import necessary libraries
import cv2
import numpy as np

# select default webcam as input
cap = cv2.VideoCapture(0)

# classify facial features which are needed
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

    #  if cameras is detected, start reading from the cam
    ret, incoming_frame = cap.read()

    face_rects = face_cascade.detectMultiScale(incoming_frame)

    # from face_rects extract the face
    (face_x, face_y, w, h) = tuple(face_rects[0])
    track_window = (face_x, face_y, w, h)

    # region of interest is the whole face rectangle. from (0,0) at top of face till end of face which is detected according to cascade
    # classifier method
    roi = incoming_frame[face_y : face_y + h, face_x : face_x + w]

    # convert roi to hsv for better color detection. because hsv allows for that
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # create histogram of roi as is explained in docs of meanshift. because it is required .
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

    # normalize for all frames
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # apply the termination criteria, i.e. if an epsilon valur or count is reached , terminate the iterative video capture
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    if ret == True:

        hsv = cv2.cvtColor(incoming_frame, cv2.COLOR_BGR2HSV)

        # calculate back propagation
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply the camshift algo
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # apply rectangle on the detected face
        x, y, w, h = track_window  # get the coordinates of face that is detected in the frame
        cv2.rectangle(incoming_frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        # determine area for blur
        blur_area = incoming_frame[y : y + h, x : x + w]
        # apply blur
        blur = cv2.GaussianBlur(blur_area, (51, 51), 0)
        # provide the blur to area to be blurred
        incoming_frame[y : y + h, x : x + w] = blur

        # start display the input frames in window named "Face Blur"
        cv2.imshow("Face Blur", incoming_frame)

        # assign the esc key as exit button and also wait 10ms before exiting
        k = cv2.waitKey(10) & 0xFF
        # if esc was pressed exit from loop. i.e. close capture
        if k == 27:
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
