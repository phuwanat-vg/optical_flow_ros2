import cv2
import numpy as np

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
cap = cv2.VideoCapture(0)

# Take first frame and find corners in it
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prevgray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(prev)
# Note the shape of the p0 variable and how we can access the data points
print(p0.shape)
print(p0[0][0][0])
print(p0[0][0][1])

while (cap.isOpened()):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # drawing
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (0,0,255), 2)
        frame = cv2.circle(frame,(int(a),int(b)),5, (0,0,255),-1)
    img = cv2.add(frame,mask)
    # Now update the previous frame and previous points
    prevgray = gray.copy()
    p0 = good_new.reshape(-1,1,2)
    cv2.imshow("image: ", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()