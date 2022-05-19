import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry


cam = PinholeCamera(640.0, 480.0, 633.935486, 375.879682, 665.439514, 269.897693)
vo = VisualOdometry(cam)

traj = np.zeros((600,500,3), dtype=np.uint8)

for img_id in range(0,1406):  #4541
    rawimg = cv2.imread('/home/copter/quad_ws/src/vo_monocular_ros2/vo_monocular_ros2/data/'+str(img_id)+'.png', 0)
    #img = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
    vo.update(rawimg, img_id)
    print(img_id)

    cur_t = vo.cur_t
    if(img_id > 2):
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0., 0., 0.
    draw_x, draw_y = int(x)+290, int(z)+90
    #true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

    cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
    #cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

    cv2.imshow('Road facing camera', rawimg)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)

cv2.imwrite('map.png', traj)
