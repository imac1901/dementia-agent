import cv2
import numpy as np
import open3d as o3d
import os
import time
from typing import Dict, List, Optional, Tuple

# Configuration
MAP_FILE = "maps/room_map.pcd"  # Path to your pre-built lidar map
CAMERA_INDEX = 0  # Webcam index (usually 0)
APRILTAG_FAMILY = "tag36h11"  # AprilTag family (tag36h11, tag25h9, etc.)
TAG_SIZE = 0.1  # Physical size of AprilTag in meters (adjust to your tags)

# Known AprilTag positions in the room (map coordinates)
# Format: {tag_id: [x, y, z]} in meters relative to lidar origin
# You'll need to measure these positions after placing tags
KNOWN_TAG_POSITIONS: Dict[int, np.ndarray] = {
    # Example: tag_id: position in room [x, y, z]
    # 0: np.array([1.0, 0.0, 1.5]),  # Tag 0 at x=1m, y=0m, z=1.5m
    # 1: np.array([0.0, 1.0, 1.5]),  # Tag 1 at x=0m, y=1m, z=1.5m
    # Add your tag positions here after calibration
}


def load_room_map(map_file: str) -> Optional[o3d.geometry.PointCloud]:
    """Load the pre-built 3D room map from lidar scan"""
    if not os.path.exists(map_file):
        print(f"Warning: Map file '{map_file}' not found. Using empty map.")
        return None
    
    map_cloud = o3d.io.read_point_cloud(map_file)
    if map_cloud.is_empty():
        print("Warning: Map is empty.")
        return None
    
    print(f"Loaded map with {len(map_cloud.points)} points")
    return map_cloud


def detect_apriltags(frame: np.ndarray) -> List[Dict]:
    """
    Detect AprilTags in the camera frame using OpenCV's ArUco module
    Returns list of detected tags with corners and IDs
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use ArUco dictionary for AprilTag 36h11
    # OpenCV's aruco module supports AprilTag dictionaries
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    
    # Try newer API first, fallback to older API
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        # Older OpenCV API
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    detections = []
    if ids is not None:
        for i, tag_id in enumerate(ids.flatten()):
            detections.append({
                'id': int(tag_id),
                'corners': corners[i][0]  # 4 corners, each [x, y]
            })
    
    return detections


def estimate_camera_pose(
    tag_detections: List[Dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate camera pose using detected AprilTags
    Returns (rotation_vector, translation_vector) or None if insufficient tags
    """
    if len(tag_detections) == 0:
        return None
    
    # Collect 3D-2D correspondences
    object_points = []  # 3D points in room coordinates
    image_points = []   # 2D points in image
    
    for tag in tag_detections:
        tag_id = tag['id']
        if tag_id not in KNOWN_TAG_POSITIONS:
            continue  # Skip unknown tags
        
        # Tag center position in room
        tag_center_3d = KNOWN_TAG_POSITIONS[tag_id]
        
        # Tag corners relative to center (assuming tag lies flat on wall)
        # We'll use the center point for now (can expand to use all 4 corners)
        corners_2d = tag['corners']
        
        # Use tag center as correspondence
        # For better accuracy, use all 4 corners with known relative positions
        image_points.append(corners_2d.mean(axis=0))  # Center of tag in image
        object_points.append(tag_center_3d)
    
    if len(object_points) < 3:
        return None  # Need at least 3 points for PnP
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # Solve PnP to get camera pose
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None
    
    return rvec, tvec


def get_camera_matrix(cap: cv2.VideoCapture) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get camera intrinsic parameters
    For now, returns a default matrix - you should calibrate your camera
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Default camera matrix (you should calibrate your camera!)
    fx = fy = width * 0.7  # Rough estimate
    cx = width / 2.0
    cy = height / 2.0
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Distortion coefficients (assume minimal distortion)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    return camera_matrix, dist_coeffs


def visualize_pose_in_map(
    map_cloud: o3d.geometry.PointCloud,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray
) -> o3d.visualization.Visualizer:
    """
    Visualize the camera position in the 3D map
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Localization", width=1280, height=720)
    
    # Add map
    vis.add_geometry(map_cloud)
    
    # Create camera coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    frame.translate(camera_position)
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(camera_rotation)
    frame.rotate(R, center=camera_position)
    
    vis.add_geometry(frame)
    
    return vis


def main():
    """Main localization loop"""
    print("Loading room map...")
    map_cloud = load_room_map(MAP_FILE)
    
    if map_cloud is None:
        print("Cannot proceed without map. Please create a map first.")
        return
    
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return
    
    # Get camera parameters
    camera_matrix, dist_coeffs = get_camera_matrix(cap)
    print("Camera initialized. Press 'q' to quit.")
    print("\nNOTE: You need to configure KNOWN_TAG_POSITIONS with your tag locations!")
    
    # Setup visualization
    vis = None
    camera_pos_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # Detect AprilTags
            tag_detections = detect_apriltags(frame)
            
            # Draw detected tags on frame
            for tag in tag_detections:
                corners = tag['corners'].astype(int)
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                tag_id = tag['id']
                center = corners.mean(axis=0).astype(int)
                cv2.putText(
                    frame, f"ID:{tag_id}", tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # Estimate camera pose
            pose = estimate_camera_pose(tag_detections, camera_matrix, dist_coeffs)
            
            if pose is not None:
                rvec, tvec = pose
                
                # Convert to position (tvec is camera position in room coordinates)
                camera_position = tvec.flatten()
                camera_pos_history.append(camera_position.copy())
                
                # Display position on frame
                pos_text = f"Position: ({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})"
                cv2.putText(
                    frame, pos_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                print(f"Camera position: {camera_position}")
                
                # Update 3D visualization
                if vis is None:
                    vis = visualize_pose_in_map(map_cloud, camera_position, rvec)
                else:
                    # Update camera frame position
                    vis.clear_geometries()
                    vis.add_geometry(map_cloud)
                    frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                    frame_vis.translate(camera_position)
                    R, _ = cv2.Rodrigues(rvec)
                    frame_vis.rotate(R, center=camera_position)
                    vis.add_geometry(frame_vis)
                    vis.poll_events()
                    vis.update_renderer()
            else:
                cv2.putText(
                    frame, "No pose (need 3+ known tags)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            # Show camera feed
            cv2.imshow("Camera Localization", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if vis is not None:
            vis.destroy_window()
        
        if camera_pos_history:
            print(f"\nLogged {len(camera_pos_history)} position estimates")


if __name__ == "__main__":
    main()

