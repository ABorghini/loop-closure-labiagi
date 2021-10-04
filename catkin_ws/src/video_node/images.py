#!/usr/bin/python
import rospy
import cv2
import os
import numpy as np
import json
import time
import math
import argparse
import rosbag
import cv_bridge
from tqdm import tqdm

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == '__main__':
    path = os.getcwd()
    bag = rosbag.Bag("april_tag_corridor_rectified_2021-05-26-14-19-27.compressed.bag", "r")
    bridge = CvBridge()
    count = 0 
    idx = 100000
    print(bag.get_type_and_topic_info())
    for topic, msg, t in tqdm(bag.read_messages(topics=["/camera/fisheye1_rect/image_raw"])):
        if count%3 == 1:    
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            os.chdir(path+"/src/video_node/images")
            cv2.imwrite("frame"+str(idx)+".jpg", cv_img)
            idx += 1
        count += 1
    bag.close()  