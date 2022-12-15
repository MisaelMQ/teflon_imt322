#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

#kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# HSV Mask (red)
lower_lim_r = np.array([0,85,120],dtype='uint8')
upper_lim_r = np.array([20,180,255],dtype='uint8')

# HSV Mask (green)
lower_lim_g = np.array([65,40,110],dtype='uint8')
upper_lim_g = np.array([100,170,185],dtype='uint8')

# HSV Mask (yellow)
lower_lim_y = np.array([15, 70,100],dtype='uint8')
upper_lim_y = np.array([50,170,255],dtype='uint8')


class Mazinger():
    def __init__(self):
        # Declaring publishers and suscribers
        self.cam_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.cam_callback)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Bridge
        self.bridge = CvBridge()

        # Twist
        self.vel = Twist()

         # Declaring variables
        self.img = np.zeros((720,1080,3), dtype="uint8")

        # Configuration and Initial States
        self.vel.linear.x = 0
        self.rate = rospy.Rate(10)  # 10hz
        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)

    def move_robot(self):
        # Speed for Movement
        aux = 0.5

        while not self.ctrl_c:
            # Getting Image
            new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype("uint8")
            hsv_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV).astype("uint8")

            # Mask Red
            mask_r = cv2.inRange(hsv_img, lower_lim_r, upper_lim_r)
            mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
            mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
            result_r = cv2.bitwise_and(new_img, new_img, mask=mask_r)

            # Mask Green
            mask_g = cv2.inRange(hsv_img, lower_lim_g, upper_lim_g)
            mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel)
            mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel)
            result_g = cv2.bitwise_and(new_img, new_img, mask=mask_g)

            # Mask Yellow
            mask_y = cv2.inRange(hsv_img, lower_lim_y, upper_lim_y)
            mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
            mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
            result_y = cv2.bitwise_and(new_img, new_img, mask=mask_y)

            # Recognizing Red
            if(np.sum(mask_r)/255 > 777600*0.1):
                MR = cv2.moments(mask_r)
                cXR = int(MR["m10"]/MR["m00"])
                cYR = int(MR["m01"]/MR["m00"])

                cv2.circle(new_img, (cXR,cYR), 5, (0,0,255), -1)
                cv2.putText(new_img, "Red", (cXR-50, cYR-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                print('Stopping ...')
                self.vel.linear.x = 0   
                print('Speed: ', round(self.vel.linear.x, 4), ' [m/s]')
            
            # Recognizing Green
            if(np.sum(mask_g)/255 > 777600*0.1):
                MG = cv2.moments(mask_g)
                cXG = int(MG["m10"]/MG["m00"])
                cYG = int(MG["m01"]/MG["m00"])
                
                cv2.circle(new_img, (cXG,cYG), 5, (0,255,0), -1)
                cv2.putText(new_img, "Green", (cXG-50, cYG-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                print('Going forward ...')
                self.vel.linear.x = aux
                print('Speed: ', round(self.vel.linear.x, 4), ' [m/s]')
            
            # Recognizing Yellow
            if(np.sum(mask_y)/255 > 777600*0.1):
                MY = cv2.moments(mask_y)
                cXY = int(MY["m10"]/MY["m00"])
                cYY = int(MY["m01"]/MY["m00"])

                cv2.circle(new_img, (cXY,cYY), 5, (0,255,255), -1)
                cv2.putText(new_img, "Yellow", (cXY-50, cYY-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,255), 3)    

                print('Going backward ...')
                self.vel.linear.x = -aux
                print('Speed: ', round(self.vel.linear.x, 4), ' [m/s]') 
            

            #cv2.imshow("Image Window", new_img)
            #cv2.imshow("Mask Red", mask_r)
            #cv2.imshow("Mask Green", mask_g)
            #cv2.imshow("Mask Yellow", mask_y)
            #cv2.waitKey(20)

            print('----------------------------------------------')
            self.vel_publisher.publish(self.vel)
            self.rate.sleep()

    def cam_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.img = cv_image.astype("uint8")
        except CvBridgeError:
            print("CvBridge Error")
            pass

    def shutdownhook(self):
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0
        self.vel_publisher.publish(self.vel)
        self.ctrl_c = True


if __name__ == '__main__':
    rospy.init_node('cam_test_3_node', anonymous=True)
    teflon_object = Mazinger()
    try:
        teflon_object.move_robot()
    except rospy.ROSInterruptException:
        pass

