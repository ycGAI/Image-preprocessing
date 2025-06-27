import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PositionDetector:
    """图像位置检测器
    
    用于检测图像是否在同一位置拍摄
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.95,
                 feature_method: str = 'orb',
                 max_features: int = 500,
                 histogram_weight: float = 0.3,
                 structural_weight: float = 0.7):
        """初始化位置检测器
        
        Args:
            similarity_threshold: 相似度阈值
            feature_method: 特征检测方法 ('orb', 'sift', 'template')
            max_features: 最大特征点数量
            histogram_weight: 直方图相似度权重
            structural_weight: 结构相似度权重
        """
        self.similarity_threshold = similarity_threshold
        self.feature_method = feature_method
        self.max_features = max_features
        self.histogram_weight = histogram_weight
        self.structural_weight = structural_weight
        
        # 初始化特征检测器
        if feature_method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif feature_method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.detector = None
            self.matcher = None
            
    def calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算直方图相似度
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            相似度分数 (0-1)
        """
        # 转换为HSV空间
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # 计算直方图
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # 归一化
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # 计算相关性
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return correlation
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算结构相似性指数 (SSIM)
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            SSIM分数 (0-1)
        """
        # 确保图像大小相同
        if img1.shape != img2.shape:
            # 调整到相同大小
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))
            
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 计算SSIM
        # 参数设置
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 计算均值
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_2 = mu1 ** 2
        mu2_2 = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_2 = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_2
        sigma2_2 = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_2
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2))
        
        return np.mean(ssim_map)
    
    def calculate_template_matching(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """使用模板匹配计算相似度
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            匹配分数 (0-1)
        """
        # 确保图像大小相同
        if img1.shape != img2.shape:
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))
            
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 使用归一化相关系数匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        return max_val
    
    def calculate_feature_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """使用特征匹配计算相似度
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            相似度分数 (0-1)
        """
        if self.detector is None:
            return 0.0
            
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.0
            
        # 匹配特征点
        matches = self.matcher.match(des1, des2)
        
        # 计算匹配分数
        if len(matches) == 0:
            return 0.0
            
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 计算好的匹配比例
        good_matches = [m for m in matches if m.distance < 50]  # ORB距离阈值
        
        match_ratio = len(good_matches) / max(len(kp1), len(kp2))
        
        return min(1.0, match_ratio * 2)  # 缩放到0-1范围
    
    def compare_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """比较两张图像的相似度
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            (总相似度, 各项指标)
        """
        metrics = {}
        
        # 计算直方图相似度
        hist_sim = self.calculate_histogram_similarity(img1, img2)
        metrics['histogram_similarity'] = float(hist_sim)
        
        # 计算结构相似度
        ssim = self.calculate_ssim(img1, img2)
        metrics['ssim'] = float(ssim)
        
        # 根据选择的方法计算额外相似度
        if self.feature_method == 'template':
            template_sim = self.calculate_template_matching(img1, img2)
            metrics['template_matching'] = float(template_sim)
            structural_sim = template_sim
        else:
            feature_sim = self.calculate_feature_similarity(img1, img2)
            metrics['feature_similarity'] = float(feature_sim)
            structural_sim = (ssim + feature_sim) / 2
            
        # 计算加权总分
        total_similarity = (self.histogram_weight * hist_sim + 
                          self.structural_weight * structural_sim)
        
        metrics['total_similarity'] = float(total_similarity)
        
        return total_similarity, metrics
    
    def detect_same_position_groups(self, image_paths: List[Path], 
                                  max_group_size: int = 10) -> List[List[Path]]:
        """检测同一位置拍摄的图像组
        
        Args:
            image_paths: 图像路径列表
            max_group_size: 最大组大小（避免内存溢出）
            
        Returns:
            同位置图像组列表
        """
        if len(image_paths) < 2:
            return []
            
        # 按文件名排序（假设按时间顺序命名）
        sorted_paths = sorted(image_paths)
        
        groups = []
        current_group = []
        reference_image = None
        
        for i, img_path in enumerate(sorted_paths):
            try:
                # 读取当前图像
                current_image = cv2.imread(str(img_path))
                if current_image is None:
                    logger.warning(f"无法读取图像: {img_path}")
                    continue
                    
                # 缩小图像以加快处理速度
                scale_factor = 0.5
                current_image = cv2.resize(current_image, None, 
                                         fx=scale_factor, fy=scale_factor)
                
                if reference_image is None or len(current_group) == 0:
                    # 开始新组
                    reference_image = current_image
                    current_group = [img_path]
                else:
                    # 与参考图像比较
                    similarity, _ = self.compare_images(reference_image, current_image)
                    
                    if similarity >= self.similarity_threshold:
                        # 添加到当前组
                        current_group.append(img_path)
                        
                        # 检查组大小限制
                        if len(current_group) >= max_group_size:
                            groups.append(current_group)
                            current_group = []
                            reference_image = None
                    else:
                        # 保存当前组并开始新组
                        if len(current_group) > 1:
                            groups.append(current_group)
                        
                        reference_image = current_image
                        current_group = [img_path]
                        
            except Exception as e:
                logger.error(f"处理图像 {img_path} 时出错: {str(e)}")
                continue
                
        # 保存最后一组
        if len(current_group) > 1:
            groups.append(current_group)
            
        return groups
    
    def is_same_position(self, img1_path: Path, img2_path: Path) -> Tuple[bool, float]:
        """判断两张图像是否在同一位置拍摄
        
        Args:
            img1_path: 第一张图像路径
            img2_path: 第二张图像路径
            
        Returns:
            (是否同位置, 相似度分数)
        """
        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                return False, 0.0
                
            similarity, _ = self.compare_images(img1, img2)
            
            return similarity >= self.similarity_threshold, similarity
            
        except Exception as e:
            logger.error(f"比较图像时出错: {str(e)}")
            return False, 0.0