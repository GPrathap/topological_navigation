#!/usr/bin/env python
"""
Created on Apr 24 22:02:24 2024
@author: Geesara Kulathunga (ggeesara@gmail.com)
"""
###################################################################################################################
import sys, json, numpy as np
import rclpy, tf2_ros, os
import yaml, copy
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup 
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor 
from rclpy import Parameter 
from ament_index_python.packages import get_package_share_directory
###################################################################################################################    

class PoleFinder(rclpy.node.Node):
    def __init__(self, name, wtags):
        super().__init__(name)
        self.declare_parameter("poles_info_yaml", Parameter.Type.STRING)
        pole_info_yaml_default = os.path.join(get_package_share_directory('topological_navigation')
                                              , 'config', 'poles.yaml')
        self.pole_info_yaml = self.get_parameter_or("poles_info_yaml", Parameter('str'
                            , Parameter.Type.STRING, pole_info_yaml_default)).value
        self.get_logger().info("Poles info are obtained from {}".format(self.pole_info_yaml))
        self.pole_poses = None
        self.pole_poses_as_mat = None
        self.pole_n_rows = None
        self.pole_thetas = None 
        self.load_poles_info()

        self.callback_localize_pose = ReentrantCallbackGroup()
        self.qos_default_sensor = QoSProfile(depth=1, 
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        
        self.estimated_pole_info = self.create_subscription(MarkerArray, '/estimated_pole_locations'
                            , self.exact_exact_pole_info, qos_profile=self.qos_default_sensor
                            , callback_group=self.callback_localize_pose)
        
        self.exact_pole_info = self.create_publisher(MarkerArray, '/exact_pole_locations'
                            , qos_profile=self.qos_default_sensor
                            , callback_group=self.callback_localize_pose)

    def load_yaml(self, filename):
        with open(filename,'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
        

    def save_yaml(self, filename, data, dfs=True):
        with open(filename,'w') as f:
            return yaml.dump(data, f, default_flow_style=dfs)
            

    def exact_exact_pole_info(self, poses):
        markerArray = MarkerArray()
        id = 0
        for marker in poses.markers:
            estimated_pose = [marker.pose.position.x, marker.pose.position.y]
            node_info = self.get_closest_pole(marker.pose)
            if(node_info is not None):
                marker = Marker()
                marker.id = id
                marker.header.frame_id = "map"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = node_info[0]
                marker.pose.position.y = node_info[1] 
                marker.pose.position.z = 1.6
                markerArray.markers.append(marker)
                id += 1
        self.exact_pole_info.publish(markerArray)

    def get_closest_pole(self, pose):
        pole_index = -1
        min_distance = 10000000
        x_map, y_map = pose.position.x, pose.position.y
        if(self.pole_poses_as_mat.shape[0]>0):
            pole_index = np.argmin(np.linalg.norm(self.pole_poses_as_mat - np.array([x_map, y_map]), axis=1))
            return self.pole_poses_as_mat[pole_index]
        return np.array([x_map, y_map])
        
    def extract_poles_info(self, poles_unsorted):
        poles = []; n_rows = []; thetas = []
        for tunnel in poles_unsorted:
            rows = tunnel[list(tunnel.keys())[0]]
            count = 0
            for i, row in enumerate(rows):
                _row = row[list(row.keys())[0]]
                
                coordinates = [[coord["x"], coord["y"]] for coord in _row["coordinates"]]
                poles.append(coordinates)
                thetas.append(_row["orientation"])
                count += 1
            n_rows.append(count)
        return poles, n_rows, thetas
    
    def load_poles_info(self, ):
        poles_info = self.load_yaml(self.pole_info_yaml)
        poles, n_rows, thetas = self.extract_poles_info(poles_info)
        self.pole_poses = list(poles)
        self.pole_poses_as_mat = np.array(self.pole_poses).reshape(-1, 2)
        self.pole_n_rows = list(n_rows)
        self.pole_thetas = list(thetas)
        
###################################################################################################################
def main(args=None):
    rclpy.init(args=args)
    wtags = True
    node = PoleFinder('pole_finder', wtags)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('shutting down localisation node\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()