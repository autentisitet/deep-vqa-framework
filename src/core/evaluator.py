from pathlib import Path
import numpy as np
import pandas as pd
import time
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger
from typing import Dict, Any, Union, List, Optional

class Evaluator:
    """
    自适应评估底座：计算学术/工业核心指标。
    完美对接顶层 main.py 的【五大子域】策略，内置 PyTorch 张量自适应脱钩剥离技术。
    """
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self, task_type: str = "iqa", model_name: str = "resnet50", version: str = "v1"):
        self.task_type = task_type
        self.model_name = model_name
        self.version = version
        self.current_date = time.strftime("%Y%m%d")

        self.logs_dir = self._PROJECT_ROOT / "results" / "train_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._base_filename = f"{self.current_date}_{task_type}_{model_name}_{version}"

        self._history_path = self.logs_dir / f"{self._base_filename}_history.csv"
        self._manifest_path = self.logs_dir / f"{self._base_filename}_manifest.csv"

        logger.info(f"⚙️  [Evaluator] 学术评估底座初始化完毕。基础标识流: {self._base_filename}")

    @property
    def base_filename(self) -> str:
        return self._base_filename

    @base_filename.setter
    def base_filename(self, new_name: str):
        """💎【精准修复 1】：当 trainer.py 动态隔离 Fold 名时，自适应同步联动重构落盘路径"""
        self._base_filename = new_name
        # 保持目录不变，仅级联刷新文件名
        if hasattr(self, '_history_path'):
            self._history_path = self.logs_dir / f"{new_name}_history.csv"
            self._manifest_path = self.logs_dir / f"{new_name}_manifest.csv"
            logger.debug(f"🔄 [Evaluator] 资产寻址路由动态重定向 -> {self._history_path.name}")

    @property
    def history_path(self) -> Path:
        return self._history_path

    @history_path.setter
    def history_path(self, path: Union[str, Path]):
        self._history_path = Path(path)
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @manifest_path.setter
    def manifest_path(self, path: Union[str, Path]):
        self._manifest_path = Path(path)
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, y_true, y_pred, epoch: int = None, train_loss: float = None, val_loss: float = None, traditional_metrics: dict = None, save_manifest: bool = False) -> dict:
        """
        计算评估指标，追加历史记录，并动态吐出对账原材料。
        """
        # 💎【精准修复 2】：利用超能力，将 y_true, y_pred 从可能包含的 PyTorch Tensor、List 中彻底无痕剥离脱钩
        y_true = self._to_clean_numpy(y_true).flatten()
        y_pred = self._to_clean_numpy(y_pred).flatten()

        # 1. 安全计算标量指标
        metrics = self._compute_metrics(y_true, y_pred)

        # 2. 揉入运行期上下文
        if epoch is not None:
            metrics['epoch'] = epoch
        if train_loss is not None:
            metrics['train_loss'] = round(float(train_loss), 6)
        if val_loss is not None:
            metrics['val_loss'] = round(float(val_loss), 6)

        # 3. 标准控制台高价值打印
        epoch_str = f"Epoch {epoch:03d} | " if epoch is not None else "Final Eval | "
        loss_str = f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | " if train_loss is not None else ""

        logger.info(
            f"📊 [{epoch_str}{self.model_name.upper()}] {loss_str}"
            f"PLCC: {metrics['plcc']:.4f} | SROCC: {metrics['srocc']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}"
        )

        self._save_history(metrics)

        # 只在需要时保存详细预测结果
        if save_manifest:
            self._save_manifest(y_true, y_pred, traditional_metrics)

        return metrics



    def evaluate(self, y_true, y_pred, epoch: int = None, train_loss: float = None, val_loss: float = None, traditional_metrics: dict = None, save_manifest: bool = False) -> dict:
        """
        🚀【架构无缝连接器】统一对接 TrainerEngine 默认调用
        """
        return self.execute(
            y_true=y_true,
            y_pred=y_pred,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            traditional_metrics=traditional_metrics,
            save_manifest=save_manifest
        )



    def _to_clean_numpy(self, data: Any) -> np.ndarray:
        """💎【精准修复 2.2】：终极防御看门狗，强行将任意 PyTorch Tensor/GPU 资产脱钩为标准 CPU NumPy"""
        if data is None:
            return np.array([])
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, list):
            # 处理 list 中包裹 Tensor 的极端情况
            return np.array([self._to_clean_numpy(x) for x in data])
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)



    def _compute_metrics(self, y_true, y_pred) -> dict:
        """计算核心质量评估指标，内置防崩溃零标准差防御"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'plcc': 0.0, 'srocc': 0.0, 'krocc': 0.0, 'rmse': 0.0, 'r2': 0.0, 'mae': 0.0}

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
        mae = np.mean(np.abs(y_true - y_pred))

        # 动态防死锁常数防御
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            plcc, srocc, krocc = 0.0, 0.0, 0.0
        else:
            plcc, _ = pearsonr(y_true, y_pred)
            srocc, _ = spearmanr(y_true, y_pred)
            krocc, _ = kendalltau(y_true, y_pred)

        return {
            'plcc': round(float(np.nan_to_num(plcc)), 6),
            'srocc': round(float(np.nan_to_num(srocc)), 6),
            'krocc': round(float(np.nan_to_num(krocc)), 6),
            'rmse': round(float(np.nan_to_num(rmse)), 6),
            'r2': round(float(np.nan_to_num(r2)), 6),
            'mae': round(float(np.nan_to_num(mae)), 6),
        }



    def _save_history(self, metrics: dict):
        """保存演进指标到历史 CSV（安全追加与 Epoch 级别的幂等去重）"""
        df_new = pd.DataFrame([metrics])

        front_cols = [c for c in ['epoch', 'train_loss', 'val_loss'] if c in df_new.columns]
        other_cols = [c for c in df_new.columns if c not in front_cols]
        df_new = df_new[front_cols + other_cols]

        if self._history_path.exists():
            try:
                df_existing = pd.read_csv(self._history_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                if 'epoch' in df_combined.columns:
                    df_combined.drop_duplicates(subset=['epoch'], keep='last', inplace=True)
                df_combined.to_csv(self._history_path, index=False)
            except Exception as e:
                logger.error(f"❌ Failed to append history CSV: {e}")
        else:
            df_new.to_csv(self._history_path, index=False)



    def _save_manifest(self, y_true, y_pred, traditional_metrics: dict = None):
        """
        将每一对样本的 true、pred 以及传统学术特征安全洗入 CSV 资产大表
        """
        manifest_data = {
            'true': y_true,
            'pred': y_pred
        }

        # 💎【精准修复 2.3】：自适应解耦传统特征字典树中的 PyTorch 批量堆叠张量
        if traditional_metrics and isinstance(traditional_metrics, dict):
            for metric_name, val_source in traditional_metrics.items():
                clean_arr = self._to_clean_numpy(val_source).flatten()
                if len(clean_arr) == len(y_true):
                    manifest_data[metric_name.lower()] = clean_arr
                else:
                    logger.debug(f"⚠️ [Evaluator] 指标 {metric_name} 长度 ({len(clean_arr)}) 与样本量 ({len(y_true)}) 不对齐，跳过对账。")

        df_manifest = pd.DataFrame(manifest_data)
        df_manifest.to_csv(self._manifest_path, index=False)
        logger.debug(f"💾 Fresh manifest alignment written to {self._manifest_path}")