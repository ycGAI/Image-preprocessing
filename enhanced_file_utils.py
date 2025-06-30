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
    """Enhanced file handling utility class
    
    Supports copy, move and symlink operations
    """
    
    def __init__(self, 
                 supported_formats: Optional[List[str]] = None,
                 operation_type: FileOperationType = 'copy'):
        """Initialize enhanced file utilities
        
        Args:
            supported_formats: List of supported image formats
            operation_type: File operation type ('copy', 'move', 'symlink')
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
        self.operation_type = operation_type

    def is_time_format(self, folder_name: str) -> bool:
        """Check if folder name is in time format
        
        Args:
            folder_name: Folder name
            
        Returns:
            Whether it's in time format
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
        """Find folders in time format
        
        Args:
            source_root: Source root directory path
            
        Returns:
            List of time folders
        """
        time_folders = []

        if not source_root.exists():
            logger.error(f"Source directory does not exist: {source_root}")
            return time_folders

        for folder in source_root.iterdir():
            if folder.is_dir() and self.is_time_format(folder.name):
                time_folders.append(folder)

        logger.info(f"Found {len(time_folders)} time folders")
        return sorted(time_folders)

    def find_image_json_pairs(self, folder: Path) -> List[Tuple[Path, Path]]:
        """Find image-JSON file pairs
        
        Args:
            folder: Folder path
            
        Returns:
            List of (image file, JSON file) tuples
        """
        pairs = []
        files = list(folder.glob('*'))
        
        # Group by filename
        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

        # Find paired files
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
        """Find image-JSON-TXT file triples
        
        Args:
            folder: Folder path
            
        Returns:
            List of (image file, JSON file, TXT file) tuples
        """
        triples = []
        files = list(folder.glob('*'))
        
        # Group by filename
        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

        # Find triple files
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
        """Read JSON file
        
        Args:
            json_path: JSON file path
            
        Returns:
            JSON data dictionary
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {json_path}: {str(e)}")
            return {}

    def write_json_file(self, json_path: Path, data: Dict):
        """Write JSON file
        
        Args:
            json_path: JSON file path
            data: Data to write
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write JSON file {json_path}: {str(e)}")
            raise

    def copy_file_safely(self, src: Path, dst: Path):
        """Safely copy file
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"Failed to copy file {src} -> {dst}: {str(e)}")
            raise

    def create_directory(self, path: Path):
        """Create directory
        
        Args:
            path: Directory path
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {str(e)}")
            raise

    def get_file_size(self, file_path: Path) -> int:
        """Get file size
        
        Args:
            file_path: File path
            
        Returns:
            File size (bytes)
        """
        try:
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to get file size {file_path}: {str(e)}")
            return 0

    def is_valid_image_format(self, file_path: Path) -> bool:
        """Check if it's a valid image format
        
        Args:
            file_path: File path
            
        Returns:
            Whether it's a valid format
        """
        return file_path.suffix.lower() in self.supported_formats
        
    def transfer_file(self, src: Path, dst: Path, operation_type: Optional[FileOperationType] = None):
        """Transfer file based on operation type
        
        Args:
            src: Source file path
            dst: Destination file path
            operation_type: Operation type (if None, use default)
        """
        if operation_type is None:
            operation_type = self.operation_type
            
        try:
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if operation_type == 'copy':
                shutil.copy2(src, dst)
                logger.debug(f"Copied file: {src} -> {dst}")
                
            elif operation_type == 'move':
                shutil.move(str(src), str(dst))
                logger.debug(f"Moved file: {src} -> {dst}")
                
            elif operation_type == 'symlink':
                # Create symlink
                if dst.exists():
                    dst.unlink()
                    
                # Use relative or absolute path
                if src.is_absolute():
                    os.symlink(src, dst)
                else:
                    # Calculate relative path
                    rel_path = os.path.relpath(src, dst.parent)
                    os.symlink(rel_path, dst)
                    
                logger.debug(f"Created symlink: {src} -> {dst}")
                
            else:
                raise ValueError(f"Unsupported operation type: {operation_type}")
                
        except Exception as e:
            logger.error(f"File operation failed {src} -> {dst}: {str(e)}")
            raise
            
    def transfer_file_pair(self, 
                          image_path: Path, 
                          json_path: Path,
                          target_folder: Path,
                          txt_path: Optional[Path] = None,
                          operation_type: Optional[FileOperationType] = None):
        """Transfer image-JSON file pair
        
        Args:
            image_path: Image file path
            json_path: JSON file path
            target_folder: Target folder
            txt_path: TXT file path (optional)
            operation_type: Operation type
        """
        # Create target folder
        self.create_directory(target_folder)
        
        # Transfer image file
        target_image = target_folder / image_path.name
        self.transfer_file(image_path, target_image, operation_type)
        
        # Transfer JSON file
        target_json = target_folder / json_path.name
        self.transfer_file(json_path, target_json, operation_type)
        
        # Transfer TXT file if exists
        if txt_path and txt_path.exists():
            target_txt = target_folder / txt_path.name
            self.transfer_file(txt_path, target_txt, operation_type)
            
    def set_operation_type(self, operation_type: FileOperationType):
        """Set default operation type
        
        Args:
            operation_type: File operation type
        """
        if operation_type not in ['copy', 'move', 'symlink']:
            raise ValueError(f"Unsupported operation type: {operation_type}")
            
        self.operation_type = operation_type
        logger.info(f"File operation type set to: {operation_type}")
        
    def cleanup_empty_directories(self, root_path: Path):
        """Clean up empty directories
        
        Args:
            root_path: Root directory path
        """
        for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
            if not dirnames and not filenames:
                try:
                    Path(dirpath).rmdir()
                    logger.debug(f"Deleted empty directory: {dirpath}")
                except Exception as e:
                    logger.warning(f"Cannot delete directory {dirpath}: {str(e)}")
                    
    def get_operation_stats(self, operation_type: FileOperationType) -> str:
        """Get operation type description
        
        Args:
            operation_type: Operation type
            
        Returns:
            Operation description
        """
        descriptions = {
            'copy': 'Copy',
            'move': 'Move',
            'symlink': 'Create Symlink'
        }
        return descriptions.get(operation_type, operation_type)
    
    def verify_symlink(self, symlink_path: Path) -> bool:
        """Verify if symlink is valid
        
        Args:
            symlink_path: Symlink path
            
        Returns:
            Whether it's valid
        """
        if not symlink_path.is_symlink():
            return False
            
        try:
            # Check if target file exists
            target = symlink_path.resolve()
            return target.exists()
        except Exception:
            return False
            
    def batch_verify_transfers(self, 
                             source_files: List[Path], 
                             target_folder: Path,
                             operation_type: FileOperationType) -> Dict[str, int]:
        """Batch verify file transfer results
        
        Args:
            source_files: List of source files
            target_folder: Target folder
            operation_type: Operation type
            
        Returns:
            Verification statistics
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
                # Move operation: source file should not exist, target file should exist
                if not src_file.exists() and target_file.exists():
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['warnings'].append(f"Move verification failed: {src_file}")
                    
            elif operation_type == 'copy':
                # Copy operation: both source and target files should exist
                if src_file.exists() and target_file.exists():
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['warnings'].append(f"Copy verification failed: {src_file}")
                    
            elif operation_type == 'symlink':
                # Symlink operation: verify link validity
                if self.verify_symlink(target_file):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['warnings'].append(f"Symlink verification failed: {target_file}")
                    
        return stats