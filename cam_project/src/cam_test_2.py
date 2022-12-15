#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import tf.transformations as transform
import tensorflow as tf
import os
from cv_bridge import CvBridge, CvBridgeError
from mod_functions import *


PARENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Objects to be Detected
lista = ['bird']
#lista = ['bird','bycycle','stop sign','giraffe','clock','person','laptop','tv']
# Model for Object Detection
PATH_MODEL = PARENT_PATH + '/data/weights.pb'
# List of Label Strings
PATH_LABELS = PARENT_PATH + '/data/label_map.pbtxt'
# Number of Classes
NUM_CLASSES = 90


# Loading a Frozen Tensorflow Model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.compat.v2.io.gfile.GFile(PATH_MODEL, 'rb') as f: 
		od_graph_def.ParseFromString(f.read())
		tf.import_graph_def(od_graph_def, name='')

# Loading Label Map
label_map = label_map_util.load_labelmap(PATH_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



class Mazinger():
    def __init__(self):
        # Declaring publishers and suscribers
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.cam_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.cam_callback)

        # Bridge
        self.bridge = CvBridge()

        # Declaring Z rotation angle
        self.rot_z = 0

         # Declaring variables
        self.img = np.zeros((720,1080,3), dtype="uint8")
        self.vel = Twist()

        # Configuration and initial states
        self.vel.linear.x = 0
        self.rate = rospy.Rate(50)  # 10hz
        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)

    def move_robot(self):
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                while not self.ctrl_c:
                    image_np = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    # Getting Data from Model
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict = {image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    image_np, classes_list, scores_list, boxes_list = draw_boxes(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        lista,
                        use_normalized_coordinates=True,
                        min_score_thresh=0.65,
                        line_thickness=8)

                    # 750000 200000 45000 +izq
                    #cv2.imshow("Object Detection", image_np)
                    #cv2.waitKey(3)

                    if(len(classes_list) > 0):
                        # Printing Detection Info
                        #print('Classes: ', classes_list[0])
                        #print('Scores: ', scores_list[0])
                        ymin,xmin,ymax,xmax = boxes_list[0]
                        #print('Ymin,Xmin,Ymax,Xmax: {},{},{},{}'.format(ymin,xmin,ymax,xmax))
                        center = (xmin + ((xmax-xmin)/2))*image_np.shape[1]
                        area = ((ymax-ymin)*image_np.shape[0])*((xmax-xmin)*image_np.shape[1])
                        #print('Center: ', center)
                        #print('Area: ', area)

                        aux = ''

                        if(area > 150000):
                            self.vel.linear.x = -0.4
                            aux += 'Backward'
                        elif(area < 100000):
                            self.vel.linear.x = 0.4
                            aux += 'Forward'
                        elif(100000 <= area <= 150000):
                            self.vel.linear.x = 0
                            aux += 'Stop'

                        limits = np.array([image_np.shape[1]*2/5,image_np.shape[1]*3/5],dtype="uint16")
                        aux += ' - '

                        if(center > limits[1]):
                            self.vel.angular.z = -0.4
                            aux += 'Right'
                        elif(center < limits[0]):
                            self.vel.angular.z = 0.4
                            aux += 'Left'
                        elif(limits[0] <= center <= limits[1]):
                            self.vel.angular.z = 0
                            aux += 'Stop'

                    
                        print(aux)
                        print('----------------------------------------------')

                    else:
                        self.vel.linear.x = 0
                        self.vel.angular.z = 0

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

