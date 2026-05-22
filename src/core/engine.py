import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional

# NOTE: Docker中torch依赖可能是旧版本，导致无法使用新API
# from torch.amp import autocast, GradScaler


class TrainerEngine:
    """
    数据驱动通用训练/验证引擎 (Data-Driven Training & Evaluation Engine) - 交差完美成品级
    职责纯粹：不绑定任何具体数据集与网络，只负责张量流控制、反向传播与状态分发。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        evaluator: Any,
        config: Dict[str, Any],
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self._validate_config(config)
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator = evaluator
        self.config = config
        self.scheduler = scheduler

        train_cfg = config.get('train', {})
        self.epochs = train_cfg.get('epochs', 30)
        self.grad_clip = train_cfg.get('grad_clip', None)
        self.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)  # 添加这行

        self.use_amp = self.config.get('system', {}).get('amp', True)
        # self.scaler = GradScaler(device_type=self.device_type, enabled=self.use_amp)
        self.scaler = GradScaler() if self.use_amp else None

        # Early Stopping 状态机初始化
        early_stop_cfg = train_cfg.get('early_stop', {})
        self.early_stop_enabled = early_stop_cfg.get('enabled', True)
        self.patience = early_stop_cfg.get('patience', 10)
        self.early_stop_monitor = early_stop_cfg.get('monitor', 'val_loss').lower() # 💎 统一在源头小写化
        self.early_stop_mode = early_stop_cfg.get('mode', 'min')

        self.early_stop_counter = 0
        self.best_early_stop_score = float('inf') if self.early_stop_mode == 'min' else float('-inf')

        # Checkpoint 状态机监控初始化
        checkpoint_cfg = train_cfg.get('checkpoint', {})
        self.checkpoint_monitor = checkpoint_cfg.get('monitor', 'val_srocc').lower() # 💎 统一在源头小写化
        self.checkpoint_mode = checkpoint_cfg.get('mode', 'max')
        self.best_checkpoint_score = float('inf') if self.checkpoint_mode == 'min' else float('-inf')

        logger.info(f"🚀 [Engine] 训练引擎初始化成功。设备: {self.device} | 监控指标: {self.checkpoint_monitor} | 自适应AMP: {self.use_amp}")


    def _validate_config(self, config: Dict[str, Any]):
        """配置强制检查器：在训练启动前阻断错误配置"""
        required_keys = {
            'train': ['epochs'],
            'logging': ['save_dir']
        }

        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"🚨 配置文件缺失一级节点: [{section}]")

            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"🚨 配置文件缺失必要参数: [{section}.{key}]，请检查 YAML 文件。")

        logger.info("✅ [System] 配置项校验通过，一切就绪。")



    def resume_training(self, checkpoint_path: str) -> int:
        """⚙️ 工业级断点续传看门狗"""
        from pathlib import Path
        ckpt_file = Path(checkpoint_path)

        if not ckpt_file.exists():
            logger.warning(f"⚠️ 未找到指定的断点文件: {checkpoint_path}，系统切回纯净初始化训练。")
            return 1

        logger.info(f"🔄 [Engine] 正在深度解码断点快照资产: {ckpt_file.name} ...")
        checkpoint = torch.load(ckpt_file, map_location=self.device)

        self.model.load_state_dict(checkpoint['state_dict'])
        logger.info("   ├─ [Model] 神经网络张量权重加载完毕。")

        if 'optimizer' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("   ├─ [Optimizer] 优化器一阶/二阶历史梯度动量接管成功。")

        if 'scheduler' in checkpoint and self.scheduler and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("   ├─ [Scheduler] 学习率控制流状态机同步对齐。")

        resume_epoch = checkpoint.get('epoch', 0) + 1

        if 'metrics' in checkpoint:
            hist_metrics = {k.lower(): v for k, v in checkpoint['metrics'].items()}
            self.best_checkpoint_score = hist_metrics.get(self.checkpoint_monitor, self.best_checkpoint_score)
            if self.early_stop_enabled:
                self.best_early_stop_score = hist_metrics.get(self.early_stop_monitor, self.best_early_stop_score)

        logger.info(f"   └─ [Time] 锁定断层成功！本次训练将自适应从第 {resume_epoch} 轮无缝续传。")
        return resume_epoch

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """运行单个 Epoch 的训练循环"""
        self.model.train()
        running_loss = 0.0

        # 获取梯度累积步数
        accum_steps = self.gradient_accumulation_steps

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for batch_idx, batch_data in enumerate(pbar):
            if isinstance(batch_data, dict):
                inputs = batch_data['data'].to(self.device)
                labels = batch_data['label'].to(self.device)
            else:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

            # 1. 前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)

            # 2. 损失计算
            with autocast(enabled=False):
                outputs_f32 = outputs.float()
                if outputs_f32.ndim > 1 and outputs_f32.size(-1) == 1:
                    outputs_f32 = outputs_f32.squeeze(-1)
                labels = labels.view_as(outputs_f32).float()
                loss_dict = self.criterion(outputs_f32, labels, model=self.model)
                total_loss = loss_dict['total_loss']

                # 关键：除以累积步数，使累积后的 loss 与正常 loss 量级一致
                total_loss = total_loss / accum_steps

            # 3. 反向传播（不立即更新）
            self.scaler.scale(total_loss).backward()

            # 4. 每 accum_steps 步更新一次
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # 梯度裁剪
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += total_loss.item() * accum_steps  # 恢复原始 loss

            # 日志输出
            log_interval = self.config.get('logging', {}).get('log_interval', 10)
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'Loss': f"{total_loss.item() * accum_steps:.4f}",
                    'MSE': f"{float(loss_dict.get('mse_loss', 0.0)):.4f}",
                    'Rank': f"{float(loss_dict.get('rank_loss', 0.0)):.4f}"
                })

        epoch_loss = running_loss / len(train_loader)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """运行单轮验证，并交由标准学术评估器计算全套对账指标"""
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_trues = []
        traditional_metrics_payload = {}

        for batch_data in tqdm(val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]"):
            if isinstance(batch_data, dict):
                inputs = batch_data['data'].to(self.device)
                labels = batch_data['label'].to(self.device)
                if 'traditional' in batch_data:
                    for k, v in batch_data['traditional'].items():
                        traditional_metrics_payload.setdefault(k, []).extend(v.numpy())
            else:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)

            with autocast(enabled=False):
                outputs_f32 = outputs.float()

                if outputs_f32.ndim > 1 and outputs_f32.size(-1) == 1:
                    outputs_f32 = outputs_f32.squeeze(-1)
                labels = labels.view_as(outputs_f32).float()

                loss_dict = self.criterion(outputs_f32, labels, model=self.model)
                total_loss = loss_dict['total_loss']

            running_loss += total_loss.item()
            all_preds.extend(outputs_f32.cpu().numpy().flatten())
            all_trues.extend(labels.cpu().numpy().flatten())

        val_loss = running_loss / len(val_loader)
        traditional_metrics = {k: np.array(v) for k, v in traditional_metrics_payload.items()} if traditional_metrics_payload else None

        eval_fn = getattr(self.evaluator, 'evaluate', None)
        if eval_fn is None:
            for alt_name in ['compute', 'compute_metrics', 'run', 'calculate']:
                alt_fn = getattr(self.evaluator, alt_name, None)
                if alt_fn is not None:
                    eval_fn = alt_fn
                    break

        if eval_fn is None:
            raise AttributeError("🚨 架构对账失败：你的 Evaluator 类里缺乏标准的评估函数路径！")

        metrics = eval_fn(
            y_true=np.array(all_trues),
            y_pred=np.array(all_preds),
            epoch=epoch,
            val_loss=val_loss,
            traditional_metrics=traditional_metrics
        )

        return metrics


    def fit(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 1):
        """主训练闭环控制体流"""
        logger.info(f"⏱️  [System] 启动全面数据自适应迭代管道。起点 Epoch: {start_epoch}")
        checkpoint_cfg = self.config.get('train', {}).get('checkpoint', {})

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            raw_metrics = self.evaluate(val_loader, epoch)

            # 全生态小写洗牌对账
            metrics = {k.lower(): v for k, v in raw_metrics.items()}

            val_loss = raw_metrics.get('val_loss', metrics.get('val_loss', 0.0))
            if val_loss == 0.0 and 'loss' in metrics:
                val_loss = metrics['loss']

            metrics['val_loss'] = val_loss
            metrics['loss'] = val_loss

            for indicator in ['srocc', 'plcc', 'rmse']:
                if indicator in metrics:
                    metrics[f'val_{indicator}'] = metrics[indicator]

            val_srocc = metrics.get('val_srocc', 0.0)
            val_plcc = metrics.get('val_plcc', 0.0)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss) # 动态传入监控指标
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"📊 [Epoch {epoch}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}")
            else:
                logger.info(f"📊 [Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}")

            # Early Stopping 拦截机制
            if self.early_stop_enabled:
                if self.early_stop_monitor in metrics:
                    score = metrics[self.early_stop_monitor]
                elif self.early_stop_monitor == 'val_loss':
                    score = val_loss
                else:
                    logger.warning(f"监控指标 '{self.early_stop_monitor}' 不存在，使用 val_loss 替代")
                    score = val_loss

                if self._is_improved(score, self.best_early_stop_score, self.early_stop_mode):
                    self.best_early_stop_score = score
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.patience:
                    logger.warning(f"🛑 [Early Stop] 监控指标 '{self.early_stop_monitor}' 连续 {self.early_stop_counter} 轮未见改善，触发防过拟合拦截！")
                    break

            # Checkpoint 模型资产落盘机制
            ckpt_score = metrics.get(self.checkpoint_monitor, 0.0)

            if self._is_improved(ckpt_score, self.best_checkpoint_score, self.checkpoint_mode):
                self.best_checkpoint_score = ckpt_score
                if checkpoint_cfg.get('save_best', True):
                    self._save_checkpoint(epoch, val_loss, metrics, is_best=True)

            if checkpoint_cfg.get('save_last', True) and epoch == self.epochs:
                self._save_checkpoint(epoch, val_loss, metrics, is_best=False)

            # 清理显存碎片
            torch.cuda.empty_cache()


    def _is_improved(self, current: float, best: float, mode: str) -> bool:
        if mode == 'min':
            return current < best
        elif mode == 'max':
            return current > best
        return False


    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, Any], is_best: bool):
        """精准控制模型资产落地，全生态物理路径绝对寻址"""
        from pathlib import Path
        import re
        import shutil

        # 💎【核心修复】：这里的 config['logging']['save_dir'] 是在 main.py 中被注入的绝对路径
        save_dir = Path(self.config.get('logging', {}).get('save_dir', './results/model_outputs'))
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cfg = self.config.get('train', {}).get('checkpoint', {})
        top_k = checkpoint_cfg.get('save_top_k', 3)

        # 💎【关键对账】：确保这里拿到的 base_filename 是已经在 main.py 里被清洗对齐过的
        base_name = self.evaluator.base_filename

        def extract_score(path: Path):
            import re
            # 修复：使用 re.escape 防止特殊字符
            # 兼容处理：正则匹配文件名中的指标数值
            match = re.search(rf"{re.escape(self.checkpoint_monitor)}(-?\d+\.\d+)", path.name.lower())
            return float(match.group(1)) if match else (-float('inf') if self.checkpoint_mode == 'max' else float('inf'))

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        if is_best:
            current_score = metrics.get(self.checkpoint_monitor, 0.0)
            # 💎【物理锁定】：直接拼成绝对路径
            pt_name = f"{base_name}_fold{self.config.get('current_fold', '1')}_best_epoch{epoch}_{self.checkpoint_monitor}{current_score:.4f}.pt"
            target_path = save_dir / pt_name

            torch.save(state, target_path)
            logger.info(f"🏆 [Checkpoint] 权重已物理隔离归档 ──> {target_path.resolve()}")

            # Top-K 空间清理逻辑保持不变
            all_best_pts = list(save_dir.glob(f"{base_name}_fold{self.config.get('current_fold', '1')}_best_epoch*.pt"))
            if len(all_best_pts) > top_k:
                reverse_flag = True if self.checkpoint_mode == 'max' else False
                all_best_pts.sort(key=extract_score, reverse=reverse_flag)
                for low_pt in all_best_pts[top_k:]:
                    low_pt.unlink()
                    logger.warning(f"🗑️  [Checkpoint] 空间防爆仓：移出旧资产 ──> {low_pt.name}")

            # 刷新最佳软链接
            all_best_pts = list(save_dir.glob(f"{base_name}_fold{self.config.get('current_fold', '1')}_best_epoch*.pt"))
            if all_best_pts:
                all_best_pts.sort(key=extract_score, reverse=(self.checkpoint_mode == 'max'))
                best_of_best = all_best_pts[0]
                standard_best_path = save_dir / f"{base_name}_fold{self.config.get('current_fold', '1')}_best.pt"
                shutil.copy(best_of_best, standard_best_path)

        else:
            pt_name = f"{base_name}_best_epoch{epoch}_{self.checkpoint_monitor}{current_score:.4f}.pt"
            target_path = save_dir / pt_name
            torch.save(state, target_path)
            logger.info(f"💾 [Checkpoint] 定期模型快照归档 ──> {target_path.resolve()}")