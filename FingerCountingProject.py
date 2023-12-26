import cv2   # importing the opencv-python library
import time  # importing time for fps calculations
import os    # importing os for listing the directory
import HandTrackingModule as htm

wCam, hCam = 640, 480     # standard camera dimensions (640*480),(1280*1024)(1920*1080)

cap = cv2.VideoCapture(0)   # Creating Video Object
# CAP_PROP_FRAME_WIDTH = 3, CAP_PROP_FRAME_HEIGHT = 4
cap.set(3, wCam)            # Setting dimensions of the camera (width and height)
cap.set(4, hCam)

folderPath = "Finger Images"
myList = os.listdir(folderPath)  # used to get list of all files in Finger Images
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')           # cv2.imread() method loads an image from the specified file
    print(f'{folderPath}/{imPath}')         # f string being used(helps embed expressions in string literal)
    overlayList.append(image)     # importing all the images from Finger Images
print(len(overlayList))           # checking whether all images are imported
pTime = 0                         # defining previous time (going to be used for fps)

detector = htm.handDetector(detectionCon=0.75)     # creating a detector

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()        # cap.read() returns a bool that will be true if the frame is received
    img = detector.findHands(img)    # returning our image after finding hands after sending in the image
    lmList = detector.findPosition(img, draw=False)  # creating landmark list
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape  # displaying height, width and number of channels
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()            # accessing the current time
    fps = 1/(cTime-pTime)          # calculating frame rate
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    # displaying the fps (fps in integers), location, font, scale, color, thickness
    # cv2.putText() method is used to draw a text string on any image

    cv2.imshow("ImageWindow", img)           # cv2.imshow() method is used to display an image in window
    cv2.waitKey(1)          # 1ms delay
