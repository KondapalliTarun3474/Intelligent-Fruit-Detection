import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxhands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxhands, 
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(frame, i)
                    self.mpDraw.draw_landmarks(frame, i, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPositions(self, frame, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # Convert normalized coordinates to pixel coordinates
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    # cv.putText(frame, str(id), (cx, cy), cv.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), thickness=1)
                    if(id==8):
                        cv.circle(frame, (cx, cy), 10, (240,32,160), thickness=-1)
                        
        return lmlist




def main():
    cTime = 0
    pTime = 0
    vid = cv.VideoCapture(-1)
    detector = handDetector()
    while True:
        isTrue, frame = vid.read()
        cv.flip(frame, 1)    
        frame = detector.findHands(frame)
        lmlist = detector.findPositions(frame)

        if len(lmlist) != 0:
            print(lmlist[6][1])


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame, str(f'FPS: {int(fps)}'), (10,30), cv.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), thickness=2)

        cv.imshow("Video", frame)
        cv.waitKey(1)








if __name__ == "__main__":
    main()
