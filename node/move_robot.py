#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class ImageConverter:
    """Class for converting the image data to OpenCV format and driving the robot."""

    def __init__(self):
        """Constructor method for initializing ROS subscribers and publishers."""
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def image_callback(self, data):
        """
        Callback method for processing image data.
        
        Args:
            data (sensor_msgs/Image): The image data received from the robot's camera.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(e)

        binary_mask = self.apply_binary_mask(cv_image, threshold=110)
        left_edge, right_edge = self.find_lane_edges(binary_mask)
        self.steer_robot(left_edge, right_edge, binary_mask.shape[1])

        cv2.imshow("Binary Mask", binary_mask)
        cv2.waitKey(3)

    def apply_binary_mask(self, frame, threshold):
        """
        Method for applying a binary mask to the input frame.

        Args:
            frame (numpy.ndarray): The input image frame.
            threshold (int): The threshold value for binarization.

        Returns:
            binary_mask (numpy.ndarray): The binary mask.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        return binary_mask

    def find_lane_edges(self, mask):
        """
        Method for finding the edges of the lane.

        Args:
            mask (numpy.ndarray): The binary mask of the frame.

        Returns:
            left_edge (int): The x-coordinate of the left edge of the lane.
            right_edge (int): The x-coordinate of the right edge of the lane.
        """
        h, w = mask.shape
        search_row = h - 2  

        black_indices = np.where(mask[search_row, :] == 0)[0]
        if len(black_indices) >= 2:
            left_edge = black_indices[0]
            right_edge = black_indices[-1]
            return left_edge, right_edge
        else:
            return None, None

    def steer_robot(self, left_edge, right_edge, frame_width):
        """
        Method for steering the robot based on the position of the lane.

        Args:
            left_edge (int): The x-coordinate of the left edge of the lane.
            right_edge (int): The x-coordinate of the right edge of the lane.
            frame_width (int): The width of the frame.
        """
        move = Twist()
        move.linear.x = 1.5
        if left_edge is not None and right_edge is not None:
            lane_center = (left_edge + right_edge) / 2
            err = lane_center - frame_width / 2
            move.angular.z = -float(err) / 30
        else:
            move.angular.z = 0
        self.cmd_vel_pub.publish(move)

def main():
    """Main method for initializing the ImageConverter object and starting the ROS node."""
    ic = ImageConverter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
