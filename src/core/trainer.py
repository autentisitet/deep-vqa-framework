import os
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import KFold
import copy

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("⚠️ Decord 未安装，视频加载将使用 OpenCV 回退")


from src.models.iqavqa_net import IQAVQANet, IQAVQALoss
from src.core.engine import TrainerEngine
from src.utils.file_loader import CaseInsensitiveAssetResolver

def worker_init_fn(worker_id):
    """防止多进程导致的数据增强随机种子冲突"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)



class VQA_IQADataset(Dataset):
    """
    通用自适应多模态数据加载器
    直接消费 DataEDA 治理后的 DataFrame 资产，内置高价值物理级防御看门狗
    """
    def __init__(self, df: pd.DataFrame, data_dir: Path, config: Dict = None, transform: Any = None, resolver: Any = None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.resolver = resolver

        self.config = config or {}
        self.num_frames = self.config.get('model', {}).get('num_frames', 8)

        self.traditional_cols = [c for c in ['ssim', 'vif', 'dlm', 'vmaf', 'niqe'] if c in self.df.columns]

        self.is_video_mode = any(str(sample).endswith(('.mp4', '.avi', '.mov', '.mkv'))
                                 for sample in self.df['sample_id'].head(10))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        default_payload = {
            'data': torch.zeros((self.num_frames, 3, 224, 224)),
            'label': torch.tensor(0.0, dtype=torch.float32)
        }
        row = self.df.iloc[idx]
        sample_id = str(row['sample_id']).strip()

        # 1. 物理层：获取 Path 和 属性
        try:
            # 现在的 resolve 返回的是 (Path, is_video, is_image)
            target_path, is_video, is_image = self.resolver.resolve_with_info(sample_id)
            str_path = str(target_path.absolute())
        except Exception as e:
            logger.error(f"❌ 寻址失败: {sample_id} | {e}")
            # 💎 调试拦截：寻址失败直接断点，检查为何 Metadata 指向的文件找不到
            if os.environ.get('DEBUG', '0') == '1':
                breakpoint()
            return default_payload

        # 2. 执行层：根据物理属性“自动分流” (不再依赖 self.is_video)
        data_tensor = None
        try:
            if is_video:
                # 视频逻辑：调用你原本的高性能 Decord 采样
                data_tensor = self._read_video_with_decord(str_path)
            elif is_image:
                # 图片逻辑：调用图像处理
                data_tensor = self._load_image_logic(str_path)
            else:
                logger.error(f"⚠️ 未知类型资产: {str_path}")
        except Exception as e:
            logger.error(f"🚨 数据读取链路崩溃: {str_path} | 原因: {e}")
            # 💎 调试拦截：读取崩溃直接断点，检查是否是文件损坏或解码器不兼容
            if os.environ.get('DEBUG', '0') == '1':
                breakpoint()

        # 3. 结果核验：这就是你要找的“隐形 None”的真相
        if data_tensor is None:
            logger.critical(f"💀 发生静默读取失败 (返回 None)! 样本: {sample_id}")
            if os.environ.get('DEBUG', '0') == '1':
                breakpoint()     # 💎 这里是拦截“读取成功但没拿到数据”的最后关卡
            return default_payload

        # 3. 组装 Payload (保持不变)
        label = torch.tensor(row.get('normalized_score', row['mos']), dtype=torch.float32)
        payload = {'data': data_tensor, 'label': label}

        if self.traditional_cols:
            payload['traditional'] = {col: torch.tensor(row[col], dtype=torch.float32) for col in self.traditional_cols}

        return payload



    def _read_video_with_decord(self, str_path: str) -> torch.Tensor:
        """
        使用 decord 进行高性能视频采样
        """
        if DECORD_AVAILABLE:
            try:
                # 使用 CPU 上下文，防止多进程下的 GPU 显存句柄死锁
                vr = VideoReader(str_path, ctx=cpu(0))
                total_frames = len(vr)
                max_frames = self.num_frames

                # 均匀采样索引 (比 range(max_frames) 更具代表性)
                if total_frames >= max_frames:
                    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
                else:
                    # 对于超短视频，直接全部读取并填充
                    indices = list(range(total_frames))

                # 核心：一次性获取所有帧，直接返回 (16, H, W, 3) 的 ndarray
                frames = vr.get_batch(indices).asnumpy()

                # 💎 规整化：直接用循环 resize 效率极低且易出错，考虑直接切片
                # 检查 frames 是否为空
                if frames.size == 0:
                    raise ValueError(f"Decord 返回了空帧序列: {str_path}")

                # 规整化：resize 到 (224, 224)
                # Decord 读取的已经是 RGB，无需 cv2.cvtColor
                resized_frames = [cv2.resize(f, (224, 224)) for f in frames]
                video_np = np.stack(resized_frames)
                data_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

                # 如果帧数不足，在时间轴上进行 padding (填充)
                if data_tensor.size(0) < max_frames:
                    padding = data_tensor[-1].unsqueeze(0).repeat(max_frames - data_tensor.size(0), 1, 1, 1)
                    data_tensor = torch.cat([data_tensor, padding], dim=0)

                return data_tensor

            except Exception as e:
                logger.error(f"🚨 [Decord] 视频解析严重故障: {str_path} | 错误: {e}")

        # 回退到 OpenCV
        return self._read_video_with_opencv(str_path)


    def _read_video_with_opencv(self, str_path: str) -> torch.Tensor:
        """OpenCV 视频读取回退方案"""
        cap = cv2.VideoCapture(str_path)
        if not cap.isOpened():
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        frames = []
        max_frames = self.num_frames

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        video_np = np.stack(frames)
        data_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

        if data_tensor.size(0) < max_frames:
            padding = data_tensor[-1].unsqueeze(0).repeat(max_frames - data_tensor.size(0), 1, 1, 1)
            data_tensor = torch.cat([data_tensor, padding], dim=0)

        return data_tensor


    def _load_image_logic(self, str_path: str) -> torch.Tensor:
        """封装好的图像加载逻辑"""
        img = cv2.imread(str_path)
        if img is None: raise ValueError("解码失败")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            return self.transform(img)

        img = cv2.resize(img, (224, 224))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0




class TrainerExecutionPipeline:
    """
    全自动模型训练调度管道 (Execution Pipeline) - 显存解耦完美闭环成品级
    """
    def __init__(self, config: Dict[str, Any], eda_df: pd.DataFrame, data_dir: Path, resolver: Any = None):
        self.config = config
        self.eda_df = eda_df
        self.data_dir = Path(data_dir).resolve()
        self.resolver = resolver

        self.task_type = config.get('task_type', 'iqa')
        if "video" in str(data_dir).lower() or self.config.get('dataset_type') == 'video':
            self.task_type = 'vqa'
        self.is_video = (config.get('task_type') == "vqa")

        self.train_cfg = config.get('train', {})
        self.batch_size = self.config.get('preprocessing', {}).get('batch_size', 16)
        self.num_workers = self.config.get('preprocessing', {}).get('num_workers', 4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # FIXME
    def execute_cross_validation(self, evaluator: Any):
        """
        执行标准 5 折交叉验证，内置调度器闭环、显存主动防御与 Top-K 文件隔离
        """
        n_splits = self.config.get('preprocessing', {}).get('k_fold', 5)
        logger.info(f"🧱 [Pipeline] 启动标准 {n_splits} 折交叉验证分割器...")

        clean_df = self.eda_df.copy()

        # 离群值过滤
        if 'is_outlier' in clean_df.columns:
            clean_df = clean_df[clean_df['is_outlier'] == False].reset_index(drop=True)
            logger.info(f"🧹 [Pipeline] 已过滤离群值。剩余样本: {len(clean_df)}")

        # 测试集隔离
        test_df = None
        if 'split' in clean_df.columns:
            active_df = clean_df[clean_df['split'].isin(['train', 'val'])].reset_index(drop=True)
            test_df = clean_df[clean_df['split'] == 'test']
            logger.info(f"🛡️ 测试集已隔离 (Size: {len(test_df)})")
        else:
            active_df = clean_df
            logger.warning("⚠️ 未检测到 'split' 列，存在数据泄露风险")

        original_base_filename = getattr(evaluator, 'base_filename', 'model')

        # 保存原始配置，用于测试集评估
        original_config = copy.deepcopy(self.config)
        for key in ['current_fold', 'model']:
            if key in original_config:
                original_config[key] = self.config.get(key)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(active_df)):
            current_fold = fold + 1
            logger.info(f"🌀 [Fold {current_fold}/{n_splits}] 开始...")

            self.config['current_fold'] = current_fold
            evaluator.base_filename = f"{original_base_filename}_fold{current_fold}"

            train_sub_df = active_df.iloc[train_idx]
            val_sub_df = active_df.iloc[val_idx]

            train_dataset = VQA_IQADataset(train_sub_df, self.data_dir, config=self.config, resolver=self.resolver)
            val_dataset = VQA_IQADataset(val_sub_df, self.data_dir, config=self.config, resolver=self.resolver)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True
            )

            model_config = self.config.get('model', {}).copy()
            model_config['num_frames'] = self.config.get('preprocessing', {}).get('num_frames', 8)
            self.config['model'] = model_config

            model = IQAVQANet(config=self.config)
            model = model.to(self.device)

            optimizer_cfg = self.config.get('train', {})
            lr = optimizer_cfg.get('lr', 1e-3)
            weight_decay = optimizer_cfg.get('weight_decay', 1e-4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            epochs = optimizer_cfg.get('epochs', 30)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            criterion = IQAVQALoss(config=self.config)

            engine = TrainerEngine(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=evaluator,
                config=self.config,
                scheduler=scheduler,
                device=self.device
            )

            engine.fit(train_loader, val_loader)
            logger.info(f"🏁 [Fold {current_fold}] 完成")

            # 变量在函数结束时会自动被垃圾回收，不一定需要手动删除
            for var in ['model', 'optimizer', 'scheduler', 'criterion', 'engine']:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()
            gc.collect()

            if self.config.get('train', {}).get('fast_run', False):
                logger.warning("⚡ 快跑模式，完成首折后退出")
                break

        # 测试集评估
        if test_df is not None and not test_df.empty:
            logger.info(f"🧪 开始测试集评估...")
            self._evaluate_test_set(test_df, evaluator, original_base_filename, n_splits, original_config)



    def _evaluate_test_set(self, test_df: pd.DataFrame, evaluator: Any, base_filename: str, n_splits: int, original_config: Dict = None):
        """测试集评估"""
        # 使用原始配置（没有 current_fold 等污染）
        config_to_use = original_config if original_config else self.config

        # 确保 model config 包含 num_frames
        model_config = config_to_use.get('model', {}).copy()
        model_config['num_frames'] = config_to_use.get('preprocessing', {}).get('num_frames', 8)
        config_to_use['model'] = model_config

        model = None

        try:
            model = IQAVQANet(config=self.config).to(self.device)
            save_dir = Path(self.config.get('logging', {}).get('save_dir', './results/model_outputs'))
            best_model_path = save_dir / f"{base_filename}_fold{n_splits}_best.pt"

            if best_model_path.exists():
                logger.info(f"💾 加载最佳模型: {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                logger.warning(f"⚠️ 未找到最佳模型 {best_model_path}，使用随机权重")

            model.eval()

            test_dataset = VQA_IQADataset(test_df, self.data_dir, config=config_to_use, resolver=self.resolver)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

            y_true_list, y_pred_list = [], []

            with torch.no_grad():
                for batch in test_loader:
                    data = batch['data'].to(self.device)
                    target = batch['label'].to(self.device)
                    output = model(data)
                    y_true_list.append(target.cpu())
                    y_pred_list.append(output.cpu())

            y_true = torch.cat(y_true_list).numpy()
            y_pred = torch.cat(y_pred_list).numpy()

            evaluator.evaluate(y_true, y_pred)

        finally:
            # 清理模型和显存
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()