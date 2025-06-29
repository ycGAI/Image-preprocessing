import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)

FileOperationType = Literal['copy', 'move', 'symlink']


class EnhancedFileUtils:
    """增强的文件处理工具类
    
    支持复制、移动和软链接操作
    """
    
    def __init__(self, 
                 supported_formats: Optional[List[str]] = None,
                 operation_type: FileOperationType = 'copy'):
        """初始化增强文件工具
        
        Args:
            supported_formats: 支持的图像格式列表
            operation_type: 文件操作类型 ('copy', 'move', 'symlink')
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
        self.operation_type = operation_type

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

    def find_image_json_pairs(self, folder: Path) -> List[Tuple[Path, Path]]:
        """查找图像-JSON文件对
        
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
        
    def transfer_file(self, src: Path, dst: Path, operation_type: Optional[FileOperationType] = None):
        """根据操作类型传输文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            operation_type: 操作类型（如果为None则使用默认）
        """
        if operation_type is None:
            operation_type = self.operation_type
            
        try:
            # 确保目标目录存在
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if operation_type == 'copy':
                shutil.copy2(src, dst)
                logger.debug(f"复制文件: {src} -> {dst}")
                
            elif operation_type == 'move':
                shutil.move(str(src), str(dst))
                logger.debug(f"移动文件: {src} -> {dst}")
                
            elif operation_type == 'symlink':
                # 创建软链接
                if dst.exists():
                    dst.unlink()
                    
                # 使用相对路径或绝对路径
                if src.is_absolute():
                    os.symlink(src, dst)
                else:
                    # 计算相对路径
                    rel_path = os.path.relpath(src, dst.parent)
                    os.symlink(rel_path, dst)
                    
                logger.debug(f"创建软链接: {src} -> {dst}")
                
            else:
                raise ValueError(f"不支持的操作类型: {operation_type}")
                
        except Exception as e:
            logger.error(f"文件操作失败 {src} -> {dst}: {str(e)}")
            raise
            
    def transfer_file_pair(self, 
                          image_path: Path, 
                          json_path: Path,
                          target_folder: Path,
                          txt_path: Optional[Path] = None,
                          operation_type: Optional[FileOperationType] = None):
        """传输图像-JSON文件对
        
        Args:
            image_path: 图像文件路径
            json_path: JSON文件路径
            target_folder: 目标文件夹
            txt_path: TXT文件路径（可选）
            operation_type: 操作类型
        """
        # 创建目标文件夹
        self.create_directory(target_folder)
        
        # 传输图像文件
        target_image = target_folder / image_path.name
        self.transfer_file(image_path, target_image, operation_type)
        
        # 传输JSON文件
        target_json = target_folder / json_path.name
        self.transfer_file(json_path, target_json, operation_type)
        
        # 如果有TXT文件，也传输
        if txt_path and txt_path.exists():
            target_txt = target_folder / txt_path.name
            self.transfer_file(txt_path, target_txt, operation_type)
            
    def set_operation_type(self, operation_type: FileOperationType):
        """设置默认操作类型
        
        Args:
            operation_type: 文件操作类型
        """
        if operation_type not in ['copy', 'move', 'symlink']:
            raise ValueError(f"不支持的操作类型: {operation_type}")
            
        self.operation_type = operation_type
        logger.info(f"文件操作类型设置为: {operation_type}")
        
    def cleanup_empty_directories(self, root_path: Path):
        """清理空目录
        
        Args:
            root_path: 根目录路径
        """
        for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
            if not dirnames and not filenames:
                try:
                    Path(dirpath).rmdir()
                    logger.debug(f"删除空目录: {dirpath}")
                except Exception as e:
                    logger.warning(f"无法删除目录 {dirpath}: {str(e)}")
                    
    def get_operation_stats(self, operation_type: FileOperationType) -> str:
        """获取操作类型的中文描述
        
        Args:
            operation_type: 操作类型
            
        Returns:
            中文描述
        """
        descriptions = {
            'copy': '复制',
            'move': '移动',
            'symlink': '创建软链接'
        }
        return descriptions.get(operation_type, operation_type)
    
    def verify_symlink(self, symlink_path: Path) -> bool:
        """验证软链接是否有效
        
        Args:
            symlink_path: 软链接路径
            
        Returns:
            是否有效
        """
        if not symlink_path.is_symlink():
            return False
            
        try:
            # 检查目标文件是否存在
            target = symlink_path.resolve()
            return target.exists()
        except Exception:
            return False
            
    def batch_verify_transfers(self, 
                             source_files: List[Path], 
                             target_folder: Path,
                             operation_type: FileOperationType) -> Dict[str, int]:
        """批量验证文件传输结果
        
        Args:
            source_files: 源文件列表
            target_folder: 目标文件夹
            operation_type: 操作类型
            
        Returns:
            验证统计
        """
        stats = {
            'total': len(source_files),
            'success': 0,
            'failed': 0,
            'warnings': []
        }
        
        for src_file in source_files:
            target_file = target_folder / src_file.name
            
            if operation_type == 'move':
                # 移动操作：源文件不应存在，目标文件应存在
                if not src_file.exists() and target_file.exists():
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['warnings'].append(f"移动验证失败: {src_file}")
                    
            elif operation_type == 'copy':
                # 复制操作：源文件和目标文件都应存在
                if src_file.exists() and target_file.exists():
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['warnings'].append(f"复制验证失败: {src_file}")
                    
            elif operation_type == 'symlink':
                # 软链接操作：验证链接有效性
                if self.verify_symlink(target_file):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['warnings'].append(f"软链接验证失败: {target_file}")
                    
        return stats