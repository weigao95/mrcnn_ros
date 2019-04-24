#! /usr/bin/env python
from mrcnn_ros.srv import *

import rospy
import cv2
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError


def main():
    rospy.wait_for_service('instance_segment')
    instance_segment = rospy.ServiceProxy('instance_segment', InstanceSegmentation)
    # Get the data
    cv_rgb = cv2.imread('/home/wei/data/mankey_pdc_data/mug/0_rgb.png', cv2.IMREAD_COLOR)

    # Build the request
    request = InstanceSegmentationRequest()
    bridge = CvBridge()
    request.rgb_image = bridge.cv2_to_imgmsg(cv_rgb, 'bgr8')

    # Call the service
    response = instance_segment(request)
    print response.object_names
    print response.bounding_boxes


if __name__ == '__main__':
    main()
