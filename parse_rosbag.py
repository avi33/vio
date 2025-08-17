import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np

bag_path = "/media/avi/5456165256163568/datasets/vio/MH_01_easy.bag"

# Topics in EuRoC
IMAGE_TOPIC = "/cam0/image_raw"
IMU_TOPIC   = "/imu0"

bag = rosbag.Bag(bag_path)
bridge = CvBridge()

# storage
images = []
imus = []

for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC, IMU_TOPIC]):
    if topic == IMAGE_TOPIC:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        timestamp = msg.header.stamp.to_sec()
        images.append((timestamp, cv_img))

    elif topic == IMU_TOPIC:
        timestamp = msg.header.stamp.to_sec()
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z])
        gyro  = np.array([msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z])
        imus.append((timestamp, accel, gyro))

bag.close()

print(f"Loaded {len(images)} images and {len(imus)} IMU samples.")
