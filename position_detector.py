import json
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PositionDetector:
    """基于GPS的位置检测器
    
    使用JSON文件中的GPS位置信息判断图像是否在同一位置拍摄
    """
    
    def __init__(self,
                 gps_distance_threshold: float = 2.0,  # 米
                 rotation_threshold: float = 0.1):      # 四元数差异阈值
        """初始化位置检测器
        
        Args:
            gps_distance_threshold: GPS距离阈值（米）
            rotation_threshold: 旋转差异阈值
        """
        self.gps_distance_threshold = gps_distance_threshold
        self.rotation_threshold = rotation_threshold
            
    def read_position_from_json(self, json_path: Path) -> Optional[Dict]:
        """从JSON文件读取位置信息
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            位置信息字典，包含translation和rotation
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取transform信息
            if 'transform' in data:
                transform = data['transform']
                if 'translation' in transform and 'rotation' in transform:
                    return {
                        'translation': transform['translation'],
                        'rotation': transform['rotation'],
                        'timestamp': data.get('header', {}).get('stamp', {})
                    }
            
            # 检查是否是扁平结构（您提供的格式）
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
            
            logger.warning(f"无法从JSON文件提取位置信息: {json_path}")
            return None
            
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_path}: {str(e)}")
            return None
    
    def calculate_gps_distance(self, pos1: Dict, pos2: Dict) -> float:
        """计算两个GPS位置之间的欧氏距离
        
        Args:
            pos1: 第一个位置的translation字典
            pos2: 第二个位置的translation字典
            
        Returns:
            距离（米）
        """
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_rotation_difference(self, rot1: Dict, rot2: Dict) -> float:
        """计算两个四元数旋转之间的差异
        
        Args:
            rot1: 第一个旋转四元数
            rot2: 第二个旋转四元数
            
        Returns:
            旋转差异（0-1）
        """
        # 提取四元数分量
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
        
        # 归一化四元数
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算四元数点积
        dot_product = np.abs(np.dot(q1, q2))
        
        # 限制在[-1, 1]范围内
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 计算角度差异
        angle_diff = 2 * np.arccos(dot_product)
        
        # 归一化到[0, 1]
        return angle_diff / np.pi
    
    def is_same_position(self, json_path1: Path, json_path2: Path) -> Tuple[bool, Dict[str, float]]:
        """通过GPS信息判断是否在同一位置
        
        Args:
            json_path1: 第一个JSON文件路径
            json_path2: 第二个JSON文件路径
            
        Returns:
            (是否同位置, 度量信息)
        """
        pos1 = self.read_position_from_json(json_path1)
        pos2 = self.read_position_from_json(json_path2)
        
        if pos1 is None or pos2 is None:
            logger.warning("GPS数据不可用，无法判断位置")
            return False, {'error': 'GPS data not available'}
        
        # 计算距离
        distance = self.calculate_gps_distance(
            pos1['translation'], 
            pos2['translation']
        )
        
        # 计算旋转差异
        rotation_diff = self.calculate_rotation_difference(
            pos1['rotation'], 
            pos2['rotation']
        )
        
        # 判断是否在同一位置
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
        """使用GPS信息检测同一位置拍摄的图像组
        
        Args:
            image_paths: 图像路径列表
            json_paths: JSON路径列表
            max_group_size: 最大组大小
            
        Returns:
            同位置图像组列表
        """
        if len(image_paths) != len(json_paths):
            logger.error("图像路径和JSON路径数量不匹配")
            return []
            
        if len(image_paths) < 2:
            return []
        
        # 读取所有位置信息
        positions = []
        valid_indices = []
        
        for i, (img_path, json_path) in enumerate(zip(image_paths, json_paths)):
            pos = self.read_position_from_json(json_path)
            if pos is not None:
                positions.append(pos)
                valid_indices.append(i)
            else:
                logger.warning(f"无法读取GPS信息: {json_path}")
        
        if len(valid_indices) < 2:
            logger.warning("有效GPS数据不足，无法进行位置分组")
            return []
        
        # 使用贪心算法分组
        groups = []
        used = set()
        
        for i, idx_i in enumerate(valid_indices):
            if i in used:
                continue
                
            current_group = [image_paths[idx_i]]
            used.add(i)
            
            # 查找与当前位置接近的所有图像
            for j, idx_j in enumerate(valid_indices[i+1:], start=i+1):
                if j in used:
                    continue
                
                # 计算距离
                distance = self.calculate_gps_distance(
                    positions[i]['translation'],
                    positions[j]['translation']
                )
                
                # 计算旋转差异
                rotation_diff = self.calculate_rotation_difference(
                    positions[i]['rotation'],
                    positions[j]['rotation']
                )
                
                # 如果足够接近，加入组
                if (distance <= self.gps_distance_threshold and 
                    rotation_diff <= self.rotation_threshold):
                    current_group.append(image_paths[idx_j])
                    used.add(j)
                    
                    # 检查组大小限制
                    if len(current_group) >= max_group_size:
                        break
            
            # 只保存包含多张图片的组
            if len(current_group) > 1:
                groups.append(current_group)
                logger.info(f"检测到同位置组：{len(current_group)}张图片")
        
        logger.info(f"总共检测到 {len(groups)} 个同位置组")
        return groups
    
    def get_position_summary(self, json_path: Path) -> Optional[str]:
        """获取位置信息摘要
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            位置信息字符串
        """
        pos = self.read_position_from_json(json_path)
        if pos is None:
            return None
            
        trans = pos['translation']
        rot = pos['rotation']
        
        return (f"Position: ({trans['x']:.2f}, {trans['y']:.2f}, {trans['z']:.2f}), "
                f"Rotation: (z={rot['z']:.3f}, w={rot['w']:.3f})")