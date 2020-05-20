#!/home/scahyawijaya/anaconda3/bin/python
import roslib
import rospy
import numpy as np

from detection_msgs.msg import Detection
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat

# Node for placing marker into rviz given detected iamge from Detector node
class RvizImageMarker():
    def __init__(self):
        # Define label2id
        self.label2id = {
            "Barack Obama": 0,
            "Avril Lavigne": 1,
            "Zhang Guorong": 2,
            "Game of thrones": 3,
            "Levi Ackerman": 4
        }
        self.marker_dict = {
            0: None,
            1: None,
            2: None,
            3: None,
            4: None
        }
        
        # Constant value from camera & laser scan calibration
        self.MIN_DEGREE = 247.5
        self.MIN_VISUAL_RGB_INDEX = 0
        self.MIN_VISUAL_DEPTH_INDEX = 340
        
        self.VISUAL_RGB_RANGE = 512
        self.VISUAL_DEGREE_RANGE = 45        
        self.VISUAL_DEPTH_RANGE = 220
        
        self.MAX_DEGREE = 292.5
        self.MAX_VISUAL_RGB_INDEX = 512
        self.MAX_VISUAL_DEPTH_INDEX = 560
        
        self.DEPTH_DATA_RANGE = 902
        
        # Define room border
        self.B_MIN_X = -3.5
        self.B_MAX_X = 5.0
        
        self.B_MIN_Y = -6.5        
        self.B_MAX_Y = -3.5
        
        # Init buffer
        self.depth_data = np.zeros(self.DEPTH_DATA_RANGE)
        self.position = Point(0.0, 0.0, 0.0)
        self.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.is_moving = False
        
        # Init publisher
        self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=0)

        # Init subscribers
        self.laser_sub = rospy.Subscriber("/vrep/scan", LaserScan, self.laser_scan_callback)
        self.detection_sub = rospy.Subscriber("/facedetector/faces", Detection, self.detection_callback)
        self.slam_pose_sub = rospy.Subscriber("/slam_out_pose", PoseStamped, self.slam_pose_callback)

    # Room classification
    def get_room(self):
        if self.position.x > self.B_MAX_X:
            return 'ROOM D'
        elif self.position.y > self.B_MAX_Y:
            return 'ROOM A'
        elif self.position.y < self.B_MIN_Y:
            return 'ROOM C'
        elif self.position.x >= self.B_MIN_X and self.position.x <= self.B_MAX_X \
            and self.position.y >= self.B_MIN_Y and self.position.y <= self.B_MAX_Y:
            return 'ROOM B'
        else:
            return 'UNKNOWN'
        
    # Position transform function
    def extract_global_position(self, cam_pos_x):        
        # Convert camera position to relative polar coordinate of robot
        depth_index = self.MIN_VISUAL_DEPTH_INDEX + int(float(cam_pos_x) * self.VISUAL_DEPTH_RANGE / self.VISUAL_RGB_RANGE)
        depth = self.depth_data[depth_index]
        if depth_index > 10:
            depth = max(max(depth, self.depth_data[depth_index - 5]), self.depth_data[depth_index - 10])
        if depth_index < 501:
            depth = max(max(depth, self.depth_data[depth_index + 5]), self.depth_data[depth_index + 10])
        degree = self.MIN_DEGREE + (float(cam_pos_x * self.VISUAL_DEGREE_RANGE) / self.VISUAL_RGB_RANGE)

        # Convert relative polar transformation to relative cartesian transformation matrix
        rel_pos_x = depth * np.cos(np.deg2rad(degree))
        rel_pos_y = depth * np.sin(np.deg2rad(degree))
        obj_vec = np.array([rel_pos_x, rel_pos_y, 0, 1])

        # Convert cartesian coordinate to absolute cartesian coordinate     
        rot_mat_robot = quat2mat([self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z])
        trans_mat_robot = compose(np.array([self.position.x, self.position.y, self.position.z]), rot_mat_robot, np.ones(3))

        # Perform transformation
        pos_x, pos_y, pos_z = np.dot(trans_mat_robot, obj_vec)[:3]
        
        return pos_x, pos_y
        
    # Marking function
    def mark_with_label(self, pos_x, pos_y, label):
        index = self.label2id[label]
        markers = []
        
        # Text label
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = index * 2
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = pos_x
        marker.pose.position.y = pos_y
        marker.pose.position.z = 1.0
        marker.text = label
        markers.append(marker)

        # Sphere pointer
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = (index * 2) + 1
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.r = 1.0 if index % 2 == 0 else 0.0
        marker.color.g = 1.0 if (index // 2) % 2 == 0 else 0.0
        marker.color.b = 1.0 if index > 2 else 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.y = 1.0
        marker.pose.position.x = pos_x
        marker.pose.position.y = pos_y
        marker.pose.position.z = 0.5
        markers.append(marker)
    
        return markers
    
    ###
    # Callback
    ###
    def slam_pose_callback(self, data):
        diff_x = abs(self.position.x - data.pose.position.x)
        diff_y = abs(self.position.y - data.pose.position.y)
        diff_oz = abs(self.orientation.z - data.pose.orientation.z)
        diff_ow = abs(self.orientation.w - data.pose.orientation.w)
        if diff_x < 5e-2 and diff_y < 5e-2 \
            and diff_oz < 5e-2 and diff_ow < 5e-2:
            self.is_moving = False
        else:
            self.is_moving = True
        
        # Update position
        self.position = data.pose.position
        self.orientation = data.pose.orientation
        
        # Update room text in Rviz
        room = self.get_room()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 50
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        marker.color.r = 0.8 if (room == 'ROOM A' or room == 'ROOM B') else 0.0
        marker.color.g = 0.8 if (room == 'ROOM D' or room == 'ROOM B') else 0.0
        marker.color.b = 0.8 if (room == 'ROOM C') else 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.position.x
        marker.pose.position.y = self.position.y
        marker.pose.position.z = 1.0
        marker.text = room
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)  
        
    def laser_scan_callback(self, data):
        x_ranges = data.ranges
        self.depth_data = np.array(x_ranges)

    def detection_callback(self, data):
        cam_pos_x = (data.x + (data.width // 2))
        label, confidence = data.label, data.confidence

        if confidence >= 70 and not self.is_moving:
            pos_x, pos_y = self.extract_global_position(cam_pos_x)
            markers = self.mark_with_label(pos_x, pos_y, label)
            
            self.marker_dict[markers[0].id] = markers
            marker_array = MarkerArray()
            for _, markers, in self.marker_dict.iteritems():   
                if markers:
                    marker_array.markers.append(markers[0])       
                    marker_array.markers.append(markers[1])
            self.marker_pub.publish(marker_array)

# Main function.    
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('rviz_image_marker')
    try:
        fd = RvizImageMarker()
        rospy.spin()    
    except rospy.ROSInterruptException: pass
