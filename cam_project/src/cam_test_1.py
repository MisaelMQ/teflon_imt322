#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Mazinger():
    def __init__(self):
        # Declaring publishers and suscribers
        self.cam_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.cam_callback)

        # Bridge
        self.bridge = CvBridge()

         # Declaring variables
        self.img = np.zeros((720,1080,3), dtype="uint8")

        self.rate = rospy.Rate(50)  # 10hz
        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)

    def move_robot(self):
        while not self.ctrl_c:
            new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            cv2.imshow("Image Window", new_img)
            cv2.waitKey(3)

            print('----------------------------------------------')
            self.rate.sleep()

    def cam_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.img = cv_image.astype("uint8")
        except CvBridgeError:
            print("CvBridge Error")

    def shutdownhook(self):
        self.rate.sleep()
        self.ctrl_c = True


if __name__ == '__main__':
    rospy.init_node('cam_proj_test', anonymous=True)
    rosbot_object = Mazinger()
    try:
        rosbot_object.move_robot()
    except rospy.ROSInterruptException:
        pass

