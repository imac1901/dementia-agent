import numpy as np
import open3d as o3d
import os
import time

# Configuration: Choose your lidar interface
LIDAR_MODE = "test"  # Options: "test", "ros", "file", "pcap"
# For ROS mode, uncomment: import rospy; from sensor_msgs.msg import PointCloud2

def get_lidar_scan_ros():
    """
    ROS subscriber - receives PointCloud2 messages
    Requires: pip install rospkg sensor-msgs
    """
    # Uncomment and adapt based on your ROS setup:
    # import rospy
    # from sensor_msgs.msg import PointCloud2
    # import sensor_msgs.point_cloud2 as pc2
    # 
    # rospy.init_node('lidar_mapper')
    # msg = rospy.wait_for_message('/velodyne_points', PointCloud2)  # Change topic name
    # points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    # return np.array(points, dtype=np.float32)
    raise NotImplementedError("ROS mode not configured. See comments in code.")


def get_lidar_scan_test():
    """
    Test mode - generates synthetic room-like point cloud
    Useful for testing without hardware
    """
    # Generate a simple room shape
    points = []
    
    # Floor (z=0)
    for x in np.linspace(-3, 3, 30):
        for y in np.linspace(-3, 3, 30):
            points.append([x, y, 0])
    
    # Walls
    # Back wall (y=-3)
    for x in np.linspace(-3, 3, 30):
        for z in np.linspace(0, 2.5, 25):
            points.append([x, -3, z])
    
    # Right wall (x=3)
    for y in np.linspace(-3, 3, 30):
        for z in np.linspace(0, 2.5, 25):
            points.append([3, y, z])
    
    # Add some noise
    points = np.array(points, dtype=np.float32)
    points += np.random.normal(0, 0.01, points.shape)
    
    time.sleep(0.1)  # Simulate scan time
    return points


def get_lidar_scan_file(scan_file: str):
    """Load a single PCD file"""
    if not os.path.exists(scan_file):
        return None
    cloud = o3d.io.read_point_cloud(scan_file)
    if cloud.is_empty():
        return None
    return np.asarray(cloud.points, dtype=np.float32)


def get_lidar_scan_pcap():
    """
    Read from PCAP file (Velodyne format)
    Requires: pip install python-pcap
    """
    # Implement based on your lidar's PCAP format
    # Example for Velodyne:
    # from velodyne_decoder import VelodyneDecoder
    # decoder = VelodyneDecoder("scan.pcap")
    # points = decoder.decode()
    # return points
    raise NotImplementedError("PCAP mode not configured. See comments in code.")


def get_next_scan():
    """Get the next lidar scan based on configured mode"""
    if LIDAR_MODE == "test":
        return get_lidar_scan_test()
    elif LIDAR_MODE == "ros":
        return get_lidar_scan_ros()
    elif LIDAR_MODE == "file":
        return get_lidar_scan_file("scan.pcd")
    elif LIDAR_MODE == "pcap":
        return get_lidar_scan_pcap()
    else:
        raise ValueError(f"Unknown LIDAR_MODE: {LIDAR_MODE}")


def main():
    """Live streaming lidar mapper"""
    global_map = o3d.geometry.PointCloud()
    scan_count = 0
    
    # Create output directory
    os.makedirs("maps", exist_ok=True)
    
    # Setup visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live Lidar Map", width=1280, height=720)
    vis.add_geometry(global_map)
    
    print(f"Starting live lidar mapping (mode: {LIDAR_MODE})")
    print("Press 'q' in the visualization window or Ctrl+C to stop")
    
    try:
        while True:
            # Get next scan
            points = get_next_scan()
            
            if points is None or len(points) == 0:
                print("No points received, waiting...")
                time.sleep(0.5)
                continue
            
            # Convert to Open3D point cloud
            scan_cloud = o3d.geometry.PointCloud()
            scan_cloud.points = o3d.utility.Vector3dVector(points)
            
            # Optional: Remove outliers
            scan_cloud, _ = scan_cloud.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            
            # Merge with global map
            if not global_map.is_empty():
                # For first scan, just add it
                global_map += scan_cloud
            else:
                global_map = scan_cloud
            
            # Downsample to keep map manageable
            global_map = global_map.voxel_down_sample(voxel_size=0.02)
            
            # Update visualization
            vis.update_geometry(global_map)
            vis.poll_events()
            vis.update_renderer()
            
            scan_count += 1
            print(f"Scan {scan_count}: {len(global_map.points)} points in map")
            
            # Save periodically (every 10 scans)
            if scan_count % 10 == 0:
                output_file = f"maps/live_map_scan_{scan_count:04d}.pcd"
                o3d.io.write_point_cloud(output_file, global_map)
                print(f"  Saved checkpoint to {output_file}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Save final map
        output_file = "maps/final_live_map.pcd"
        o3d.io.write_point_cloud(output_file, global_map)
        print(f"Saved final map ({len(global_map.points)} points) to {output_file}")
        vis.destroy_window()


if __name__ == "__main__":
    main()