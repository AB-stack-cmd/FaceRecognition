import cv2 as cv
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
draw_hand =  mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

ctime = 0
ptime = 0


video = cv.VideoCapture(0)

while True:
    ret , frame =  video.read()
    if ret :

        rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for handmarks in result.multi_hand_landmarks:
                 draw_hand.draw_landmarks(frame,handmarks,mp_hands.HAND_CONNECTIONS,draw_hand.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),draw_hand.DrawingSpec(color=(0,255,0), thickness=2))
        
        ctime = time.time()
        fps = 1/(ctime-ptime) if (ctime - ptime) > 0 else 0
        # when ever the frame change it will update thevalue for calcutation
        ptime = ctime

        cv.putText(frame,f"fps:{int(fps)}",(15,40),cv.FONT_HERSHEY_COMPLEX,1,color=(0,255,0))

        cv.imshow("tracker",frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv.destroyAllWindows()