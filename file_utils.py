import re
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """文件处理工具类"""

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """初始化文件工具
        
        Args:
            supported_formats: 支持的图像格式列表
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_formats = [fmt.lower() for fmt in supported_formats]

    def is_time_format(self, folder_name: str) -> bool:
        """检查文件夹名是否为时间格式
        
        Args:
            folder_name: 文件夹名称
            
        Returns:
            是否为时间格式
        """
        time_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # 2024-01-01
            r'^\d{4}_\d{2}_\d{2}$',  # 2024_01_01
            r'^\d{8}$',              # 20240101
            r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$',  # 2024-01-01_12-30-45
            r'^\d{14}$',             # 20240101123045
        ]

        for pattern in time_patterns:
            if re.match(pattern, folder_name):
                return True
        return False

    def find_time_folders(self, source_root: Path) -> List[Path]:
        """查找时间格式的文件夹
        
        Args:
            source_root: 源目录路径
            
        Returns:
            时间文件夹列表
        """
        time_folders = []

        if not source_root.exists():
            logger.error(f"源目录不存在: {source_root}")
            return time_folders

        for folder in source_root.iterdir():
            if folder.is_dir() and self.is_time_format(folder.name):
                time_folders.append(folder)

        logger.info(f"找到 {len(time_folders)} 个时间文件夹")
        return sorted(time_folders)

    def find_image_json_txt_triples(self, folder: Path) -> List[Tuple[Path, Path, Path]]:
        """查找图像-JSON-TXT文件三元组
        
        Args:
            folder: 文件夹路径
            
        Returns:
            (图像文件, JSON文件)元组列表
        """
        pairs = []
        files = list(folder.glob('*'))
        
        # 按文件名分组
        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

        # 查找配对文件
        for stem, file_list in file_groups.items():
            image_file = None
            json_file = None

            for file in file_list:
                ext = file.suffix.lower()
                if ext in self.supported_formats:
                    image_file = file
                elif ext == '.json':
                    json_file = file

            if image_file and json_file:
                pairs.append((image_file, json_file))

        return pairs

    def find_image_json_txt_triples(self, folder: Path) -> List[Tuple[Path, Path, Path]]:
        """查找图像-JSON-TXT文件三元组
        
        Args:
            folder: 文件夹路径
            
        Returns:
            (图像文件, JSON文件, TXT文件)元组列表
        """
        triples = []
        files = list(folder.glob('*'))
        
        # 按文件名分组
        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

        # 查找三元组文件
        for stem, file_list in file_groups.items():
            image_file = None
            json_file = None
            txt_file = None

            for file in file_list:
                ext = file.suffix.lower()
                if ext in self.supported_formats:
                    image_file = file
                elif ext == '.json':
                    json_file = file
                elif ext == '.txt':
                    txt_file = file

            if image_file and json_file and txt_file:
                triples.append((image_file, json_file, txt_file))

        return triples

    def read_json_file(self, json_path: Path) -> Dict:
        """读取JSON文件
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            JSON数据字典
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_path}: {str(e)}")
            return {}

    def write_json_file(self, json_path: Path, data: Dict):
        """写入JSON文件
        
        Args:
            json_path: JSON文件路径
            data: 要写入的数据
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"写入JSON文件失败 {json_path}: {str(e)}")
            raise

    def copy_file_safely(self, src: Path, dst: Path):
        """安全地复制文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
        """
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"复制文件失败 {src} -> {dst}: {str(e)}")
            raise

    def create_directory(self, path: Path):
        """创建目录
        
        Args:
            path: 目录路径
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建目录失败 {path}: {str(e)}")
            raise

    def get_file_size(self, file_path: Path) -> int:
        """获取文件大小
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件大小（字节）
        """
        try:
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"获取文件大小失败 {file_path}: {str(e)}")
            return 0

    def is_valid_image_format(self, file_path: Path) -> bool:
        """检查是否为有效的图像格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为有效格式
        """
        return file_path.suffix.lower() in self.supported_formats