import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Range, Image
from nav_msgs.msg import Odometry
from rclpy import qos
import numpy as np
import cv2
import os
import socket
import sys
from cv_bridge import CvBridge
from rclpy.time import Time

class Optical_flow(Node):
    def __init__(self):
        super().__init__("optical_flow_node")
        self.imu_sub = self.create_subscription(Imu, "imu/raw", self.imu_callback, qos_profile=qos.qos_profile_sensor_data)
        self.range_sub = self.create_subscription(Range, "lidar_altitude", self.range_callback, qos_profile=qos.qos_profile_sensor_data)
        self.odom_flow_pub = self.create_publisher(Odometry, "flow/odom", qos_profile = qos.qos_profile_sensor_data)
        self.image_flow_pub = self.create_publisher(Image, "flow/image", qos_profile = qos.qos_profile_system_default)
        self.imu_pub = self.create_publisher(Imu, "imu/data", qos_profile = qos.qos_profile_sensor_data)
        self.timer1 = self.create_timer(0.025, self.timer1_callback)
        #self.image_sub = self.create_subscription(Image, "flow_camera/image_raw", self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        self.imu_sub
        
        self.imu = Imu()
        self.of = Odometry()
        self.lidar_range = Range()
        self.image_flow = Image()
        
        #optical flow
        self.opt_velx = 0.0
        self.opt_vely = 0.0
        self.est_velx = 0.0
        self.est_vely = 0.0
        self.fps = 30.0
        self.focal = 4.0 #mm
        self.pix_size = 0.000006 #m cmos:4mm*2.25mm(mujidao @caide) 4/640 = 0.00000625
        self.cam = cv2.VideoCapture(0)
        self.ret, self.prev = self.cam.read()
        self.ret, self.ori = self.cam.read()
        self.prev = cv2.resize(self.prev, (180,120), interpolation = cv2.INTER_AREA)
        self.prevgray = cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)
        self.show_hsv = False
        self.show_glitch = False
        self.cur_glitch = self.prev.copy()
        
        self.flow = None
        
        self.v = {"x": 0.0, "y": 0.0}
        
        
        self.bridge = CvBridge()
        
    def timer1_callback(self):
        self.odom_flow_pub.publish(self.of)
    def imu_callback(self,msg):
        self.imu = msg
        self.imu.header.stamp = self.get_clock().now().to_msg()
        self.imu.header.frame_id = "imu_link"
        
        self.imu_pub.publish(self.imu)
        
    def range_callback(self,msg):
        self.lidar_range = msg
        self.lidar_range.header.stamp = self.get_clock().now().to_msg()
        self.lidar_range.header.frame_id = "rangefinder_link"
        self.flow_process()
    
    def image_callback(self, msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)
    
    def estimate_velocity(self,flowx,flowy):
        alt = self.lidar_range.range
        imu_x = self.imu.angular_velocity.x
        imu_y = self.imu.angular_velocity.y
        focal = self.focal
        
        opt_velx = flowx * self.fps * self.pix_size
        opt_vely = flowy * self.fps * self.pix_size
        est_velx = -alt*(opt_velx/focal) - (alt* imu_y)
        est_vely = -alt*(opt_vely/focal) + (alt* imu_x)
        return est_velx, est_vely
    
    def draw_flow(self,img, flow, step=16):
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
        vx,vy = self.estimate_velocity(lx,ly)
        self.v["vx"] = vx
        self.v["vy"] = vy
        self.flow_odom()
        print("vx: {} vy: {}".format(vx,vy))
        #print("FLow: ", txStr)
        
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        
        try:
            self.image_flow_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        except CvBridgeError as e:
            print(e)
        return vis
    
    def flow_odom(self):
        self.of.pose.pose.orientation.w = self.imu.orientation.w
        self.of.pose.pose.orientation.x = self.imu.orientation.x
        self.of.pose.pose.orientation.y = self.imu.orientation.y
        self.of.pose.pose.orientation.z = self.imu.orientation.z
        self.of.twist.twist.linear.x = self.v["vx"]
        self.of.twist.twist.linear.y = self.v["vy"]
        self.of.header.stamp = self.get_clock().now().to_msg()
        self.of.header.frame_id = "flow_link"
        self.of.child_frame_id = "base_link"
        
    
    def flow_process(self):
        ret, img = self.cam.read()
        img = cv2.resize(img, (180,120), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3),np.uint8)
        gray = cv2.erode(gray,kernel, iterations = 2)
        self.flow = cv2.calcOpticalFlowFarneback(self.prevgray, gray,self.flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print flow
        self.prevgray = gray
        #exit()
        
        cv2.imshow('flow', self.draw_flow(gray, self.flow))
        
        ch = 0xFF & cv2.waitKey(1)

def main():
    rclpy.init()
    of = Optical_flow()
    rclpy.spin(of)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()    
    cv2.destroyAllWindows()