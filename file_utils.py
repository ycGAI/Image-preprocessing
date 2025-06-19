import re
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FileUtils:

    def __init__(self, supported_formats: Optional[List[str]] = None):

        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_formats = [fmt.lower() for fmt in supported_formats]

    def is_time_format(self, folder_name: str) -> bool:

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

        time_folders = []

        if not source_root.exists():
            logger.error(f"not exist {source_root}")
            return time_folders

        for folder in source_root.iterdir():
            if folder.is_dir() and self.is_time_format(folder.name):
                time_folders.append(folder)

        logger.info(f"found {len(time_folders)} time folders ")
        return sorted(time_folders)

    def find_image_json_txt_triples(self, folder: Path) -> List[Tuple[Path, Path, Path]]:

        pairs = []
        files = list(folder.glob('*'))

        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

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

        triples = []
        files = list(folder.glob('*'))

        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

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

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"fail {json_path}: {str(e)}")
            return {}

    def write_json_file(self, json_path: Path, data: Dict):

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"fail {json_path}: {str(e)}")
            raise

    def copy_file_safely(self, src: Path, dst: Path):

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"fail {src} -> {dst}: {str(e)}")
            raise

    def create_directory(self, path: Path):

        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"fail {path}: {str(e)}")
            raise

    def get_file_size(self, file_path: Path) -> int:

        try:
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"fail {file_path}: {str(e)}")
            return 0

    def is_valid_image_format(self, file_path: Path) -> bool:

        return file_path.suffix.lower() in self.supported_formats