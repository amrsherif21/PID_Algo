#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import sys

class LineFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("Image window",cv_image)
        cv.waitKey(3)
        rate = rospy.Rate(2)
        move = Twist()
        move.linear.x = 0.5
        move.angular.z = 0.5
        # hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # lower_blue = np.array([110, 50, 50])
        # upper_blue = np.array([130, 255, 255])
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # h, w, d = cv_image.shape
        # search_top = 3*h/4
        # search_bot = search_top + 20
        # mask[0:int(search_top), 0:w] = 0
        # mask[int(search_bot):h, 0:w] = 0
        # M = cv2.moments(mask)
        # if M['m00'] > 0:
        #     cx = int(M['m10']/M['m00'])
        #     cy = int(M['m01']/M['m00'])
        #     err = cx - w/2
        #     self.twist.linear.x = 0.2
        #     self.twist.angular.z = -float(err) / 100
        #     self.cmd_vel_pub.publish(self.twist)

    def main(args):
        rospy.init_node('line_follower',anonymous=True)
        line_follower = LineFollower()
        try"
            rospy.spin()
        except KeyboardInterrupt:
            print("Off")
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main(sys.argv)
