import cv2
import mediapipe as mp     # importing mediapipe library as mp
import time

cap = cv2.VideoCapture(0)
# importations and initializations
mpHands = mp.solutions.hands  # formality to work with this model
hands = mpHands.Hands()   # uses RGB images # object creation (hands) # working with default parameters
mpDraw = mp.solutions.drawing_utils   # method provided by mediapipe for the 21 points

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # .cvtColor() method is used to convert an image from one color space to another
    results = hands.process(imgRGB)
    # Processes an RGB image and returns the hand landmarks and handedness of each detected hand.
    # print(results.multi_hand_landmarks)   # to check whether something is detected or not

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # for checking number of hands # extracting information of each hand
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)          # printing the id number and landmark
                h, w, c = img.shape      # getting the dimensions of img window
                cx, cy = int(lm.x*w), int(lm.y*h)
                # multiplying the x landmark pixel with width and y landmark pixel with the height to get cx, cy
                print(id, cx, cy)
                # if id == 4:    # detection according to the index given to point on hand(4 for tip of thumb)
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                # at the bottom of wrist would draw a circle on the img at cx, cy 15 radius purple color and filled
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # Hand connections method provided by mediapipe to show connections between the 21 point of hand

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
