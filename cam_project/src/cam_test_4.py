#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import pytesseract
from cv_bridge import CvBridge, CvBridgeError


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
        aux = 0.2

        while not self.ctrl_c:
            # Getting Image
            new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype("uint8")
            gray_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY).astype("uint8")
           
            # Gaussian Blur
            kernel_size = 7
            gaussian = cv2.GaussianBlur(gray_img, (kernel_size,kernel_size), 0, 0)
            ret, thresh = cv2.threshold(gaussian, 150, 200, cv2.THRESH_BINARY_INV)

            # Morphological Operations
            filtered = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))

            # Getting Text
            text = pytesseract.image_to_string(filtered, config='--psm 11')

            # Conditions for Movement
            if('AVANZAR' in text):
                print('Text: ', text)
                print('Going forward ...')
                self.vel.linear.x = aux
            if('RETROCEDER' in text):
                print('Text: ', text)
                print('Going backward ...')
                self.vel.linear.x = -aux
            if('PARAR' in text):
                print('Text: ', text)
                print('Stopping ...')
                self.vel.linear.x = 0 
         
            #cv2.imshow("Image Window", new_img)
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
    rospy.init_node('cam_proj_test', anonymous=True)
    rosbot_object = Mazinger()
    try:
        rosbot_object.move_robot()
    except rospy.ROSInterruptException:
        pass

