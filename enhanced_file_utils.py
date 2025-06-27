import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Literal
import logging

from file_utils import FileUtils

logger = logging.getLogger(__name__)


FileOperationType = Literal['copy', 'move', 'symlink']


class EnhancedFileUtils(FileUtils):
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
        super().__init__(supported_formats)
        self.operation_type = operation_type
        
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