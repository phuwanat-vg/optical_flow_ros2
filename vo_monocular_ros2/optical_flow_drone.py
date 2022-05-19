import numpy as np
import cv2
import os
import socket

opt_velx = 0.0
opt_vely = 0.0
est_velx = 0.0
est_vely = 0.0
fps = 30.0
focal  = 4 #mm
pix_size = 0.000006 #m cmos:4mm*2.25mm(mujidao @caide) 4/640 = 0.00000625  
imu_y = 0.1
imu_x = 0.1
alt = 2.0


def estimate_velocity(flowx,flowy):
    opt_velx = flowx * fps * pix_size
    opt_vely = flowy * fps * pix_size
    est_velx = alt*(opt_velx/focal) - (alt* imu_y/1000)
    est_vely = alt*(opt_vely/focal) + (alt* imu_x/1000)
    
    return est_velx, est_vely


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    y =y.astype(int)
    x =x.astype(int)
    fx, fy = flow[y,x].T
    lx =0
    ly = 0

    os.system('cls' if os.name == 'nt' else 'clear')
    for tx in fx:
        lx = tx+lx
    lx = lx/300

    for ty in fy:
        ly = ty+ly
    ly=ly/300

    txStr = "FLOW,"+str(lx)+","+str(ly)
    vx,vy = estimate_velocity(lx,ly)
    print("vx: {} vy: {}".format(vx,vy))
    print("FLow: ", txStr)
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

if __name__ == '__main__':
    import sys
    cam = cv2.VideoCapture(0)
    ret, prev = cam.read()
    ret, ori = cam.read()
    prev = cv2.resize(prev, (180,120), interpolation = cv2.INTER_AREA)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (180,120), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3),np.uint8)
        gray = cv2.erode(gray,kernel, iterations = 2)
        flow = None

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray,flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print flow
        prevgray = gray
        #exit()

        cv2.imshow('flow', draw_flow(gray, flow))
        
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        
    cv2.destroyAllWindows()