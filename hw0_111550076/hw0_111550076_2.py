import cv2
import numpy as np

video_path = 'data/video.mp4'
cap = cv2.VideoCapture(video_path)

x1, f1 = cap.read()
x2, f2 = cap.read()
cap.release()

if not x1 or not x2:
    print("fail")
else:
    diff = cv2.absdiff(f1, f2)
    #diff[:,:,0] = np.zeros([diff.shape[0], diff.shape[1]])
    #diff[:,:,2] = np.zeros([diff.shape[0], diff.shape[1]])
    diff[:,:,[0,2]] = 0
    res = np.hstack([f1, diff])

    cv2.imwrite("hw0_111550076_2.png", res)
    #cv2.imwrite("tmp.png", res)


cv2.destroyAllWindows()
