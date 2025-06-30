import json
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PositionDetector:
    """GPS-based position detector
    
    Uses GPS position information from JSON files to determine if images were taken at the same location
    """
    
    def __init__(self,
                 gps_distance_threshold: float = 2.0,  # meters
                 rotation_threshold: float = 0.1):      # quaternion difference threshold
        """Initialize position detector
        
        Args:
            gps_distance_threshold: GPS distance threshold (meters)
            rotation_threshold: Rotation difference threshold
        """
        self.gps_distance_threshold = gps_distance_threshold
        self.rotation_threshold = rotation_threshold
            
    def read_position_from_json(self, json_path: Path) -> Optional[Dict]:
        """Read position information from JSON file
        
        Args:
            json_path: JSON file path
            
        Returns:
            Position information dictionary containing translation and rotation
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract transform information
            if 'transform' in data:
                transform = data['transform']
                if 'translation' in transform and 'rotation' in transform:
                    return {
                        'translation': transform['translation'],
                        'rotation': transform['rotation'],
                        'timestamp': data.get('header', {}).get('stamp', {})
                    }
            
            # Check if it's a flat structure (format you provided)
            if 'transformtranslationx' in data:
                translation = {
                    'x': data.get('transformtranslationx'),
                    'y': data.get('transformtranslationy'),
                    'z': data.get('transformtranslationz', 0.5)
                }
                rotation = {
                    'x': data.get('transformrotationx', 0),
                    'y': data.get('transformrotationy', 0),
                    'z': data.get('transformrotationz', 0),
                    'w': data.get('transformrotationw', 1)
                }
                
                return {
                    'translation': translation,
                    'rotation': rotation,
                    'timestamp': {
                        'sec': data.get('headerstampsec', 0),
                        'nanosec': data.get('headerstampnanosec', 0)
                    }
                }
            
            logger.warning(f"Cannot extract position information from JSON file: {json_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to read JSON file {json_path}: {str(e)}")
            return None
    
    def calculate_gps_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate Euclidean distance between two GPS positions
        
        Args:
            pos1: First position's translation dictionary
            pos2: Second position's translation dictionary
            
        Returns:
            Distance (meters)
        """
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_rotation_difference(self, rot1: Dict, rot2: Dict) -> float:
        """Calculate difference between two quaternion rotations
        
        Args:
            rot1: First rotation quaternion
            rot2: Second rotation quaternion
            
        Returns:
            Rotation difference (0-1)
        """
        # Extract quaternion components
        q1 = np.array([
            rot1.get('x', 0), 
            rot1.get('y', 0), 
            rot1.get('z', 0), 
            rot1.get('w', 1)
        ])
        q2 = np.array([
            rot2.get('x', 0), 
            rot2.get('y', 0), 
            rot2.get('z', 0), 
            rot2.get('w', 1)
        ])
        
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate quaternion dot product
        dot_product = np.abs(np.dot(q1, q2))
        
        # Limit to [-1, 1] range
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle difference
        angle_diff = 2 * np.arccos(dot_product)
        
        # Normalize to [0, 1]
        return angle_diff / np.pi
    
    def is_same_position(self, json_path1: Path, json_path2: Path) -> Tuple[bool, Dict[str, float]]:
        """Determine if at the same position using GPS information
        
        Args:
            json_path1: First JSON file path
            json_path2: Second JSON file path
            
        Returns:
            (Is same position, metrics)
        """
        pos1 = self.read_position_from_json(json_path1)
        pos2 = self.read_position_from_json(json_path2)
        
        if pos1 is None or pos2 is None:
            logger.warning("GPS data not available, cannot determine position")
            return False, {'error': 'GPS data not available'}
        
        # Calculate distance
        distance = self.calculate_gps_distance(
            pos1['translation'], 
            pos2['translation']
        )
        
        # Calculate rotation difference
        rotation_diff = self.calculate_rotation_difference(
            pos1['rotation'], 
            pos2['rotation']
        )
        
        # Determine if at same position
        is_same = (distance <= self.gps_distance_threshold and 
                  rotation_diff <= self.rotation_threshold)
        
        metrics = {
            'gps_distance': float(distance),
            'rotation_difference': float(rotation_diff),
            'distance_threshold': self.gps_distance_threshold,
            'rotation_threshold': self.rotation_threshold,
            'method': 'gps'
        }
        
        return is_same, metrics
    
    def detect_same_position_groups(self, 
                                  image_paths: List[Path],
                                  json_paths: List[Path],
                                  max_group_size: int = 10) -> List[List[Path]]:
        """Detect groups of images taken at the same position using GPS information
        
        Args:
            image_paths: List of image paths
            json_paths: List of JSON paths
            max_group_size: Maximum group size
            
        Returns:
            List of same position image groups
        """
        if len(image_paths) != len(json_paths):
            logger.error("Number of image paths and JSON paths don't match")
            return []
            
        if len(image_paths) < 2:
            return []
        
        # Read all position information
        positions = []
        valid_indices = []
        
        for i, (img_path, json_path) in enumerate(zip(image_paths, json_paths)):
            pos = self.read_position_from_json(json_path)
            if pos is not None:
                positions.append(pos)
                valid_indices.append(i)
            else:
                logger.warning(f"Cannot read GPS information: {json_path}")
        
        if len(valid_indices) < 2:
            logger.warning("Insufficient valid GPS data, cannot perform position grouping")
            return []
        
        # Use greedy algorithm for grouping
        groups = []
        used = set()
        
        for i, idx_i in enumerate(valid_indices):
            if i in used:
                continue
                
            current_group = [image_paths[idx_i]]
            used.add(i)
            
            # Find all images close to current position
            for j, idx_j in enumerate(valid_indices[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Calculate distance
                distance = self.calculate_gps_distance(
                    positions[i]['translation'],
                    positions[j]['translation']
                )
                
                # Calculate rotation difference
                rotation_diff = self.calculate_rotation_difference(
                    positions[i]['rotation'],
                    positions[j]['rotation']
                )
                
                # If close enough, add to group
                if (distance <= self.gps_distance_threshold and 
                    rotation_diff <= self.rotation_threshold):
                    current_group.append(image_paths[idx_j])
                    used.add(j)
                    
                    # Check group size limit
                    if len(current_group) >= max_group_size:
                        break
            
            # Only save groups with multiple images
            if len(current_group) > 1:
                groups.append(current_group)
                logger.info(f"Detected same position group: {len(current_group)} images")
        
        logger.info(f"Total {len(groups)} same position groups detected")
        return groups
    
    def get_position_summary(self, json_path: Path) -> Optional[str]:
        """Get position information summary
        
        Args:
            json_path: JSON file path
            
        Returns:
            Position information string
        """
        pos = self.read_position_from_json(json_path)
        if pos is None:
            return None
            
        trans = pos['translation']
        rot = pos['rotation']
        
        return (f"Position: ({trans['x']:.2f}, {trans['y']:.2f}, {trans['z']:.2f}), "
                f"Rotation: (z={rot['z']:.3f}, w={rot['w']:.3f})")