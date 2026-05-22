# src/utils/file_loader.py
from pathlib import Path
from typing import Dict, Union, Set, Tuple
from loguru import logger
from dataclasses import dataclass
from src.data.types import DatasetType

@dataclass
class AssetInfo:
    """资产信息 - 单一数据源"""
    path: Path
    filename: str
    extension: str

    @property
    def dataset_type(self) -> DatasetType:
        """根据扩展名返回数据集类型"""
        return DatasetType.from_extension(self.extension)

    @property
    def is_video(self) -> bool:
        return self.dataset_type == DatasetType.VIDEO

    @property
    def is_image(self) -> bool:
        return self.dataset_type == DatasetType.IMAGE



class CaseInsensitiveAssetResolver:
    """
    工业级寻址网关：采用预构建哈希索引，彻底解决多进程 IO 竞争
    """
    def __init__(self, target_dir: Union[str, Path], allowed_extensions: list):
        self.target_dir = Path(target_dir).resolve()
        self.allowed_exts: Set[str] = {f".{ext.lower().lstrip('.')}" for ext in allowed_extensions}

        # 视频扩展名集合，用于在索引时自动判断属性
        self.video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        self.image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # 【修改点】：不再使用动态查找，而是在初始化阶段一次性完成索引
        # 这样在训练阶段，每个进程查的都是内存里的同一个表，无需再操作磁盘
        self.full_registry: Dict[str, AssetInfo] = {}

        logger.info(f"🚀 [Resolver] 正在构建全局路径索引，锚点: {self.target_dir}")

        # 只扫描一次，将所有文件路径映射到内存字典中
        # 无论嵌套多深，rglob 都能处理
        for f in self.target_dir.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower()
                if ext in self.allowed_exts:
                    # 创建 AssetInfo 对象
                    asset_info = AssetInfo(
                        path=f,
                        filename=f.name,
                        extension=ext
                    )
                    # 存入完整的物理信息
                    self.full_registry[f.name.lower()] = asset_info
                else:
                    # 如果不需要的文件，直接跳过，什么都不做
                    continue
        logger.info(f"✅ [Resolver] 索引构建完毕，已缓存 {len(self.full_registry)} 个资产。")


    def resolve(self, label_filename: str) -> Path:
        name_lower = Path(label_filename).name.lower()

        # 【修改点】：直接查字典，O(1) 复杂度，无 IO 竞争
        if name_lower in self.full_registry:
            return self.full_registry[name_lower]       # 返回 (Path, is_video)
        # 找不到时才抛异常
        raise FileNotFoundError(f"❌ 资产 '{label_filename}' 在 {self.target_dir} 中无法找到。")


    def resolve_path(self, label_filename: str) -> Path:
        """只返回路径（兼容旧代码）"""
        return self.resolve(label_filename).path


    def resolve_with_info(self, label_filename: str) -> Tuple[Path, bool, bool]:
        """兼容旧接口：返回 (path, is_video, is_image)"""
        asset = self.resolve(label_filename)
        return asset.path, asset.is_video, asset.is_image