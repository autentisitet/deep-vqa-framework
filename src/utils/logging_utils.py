# src/utils/logging_utils.py
import atexit
import sys
import time
from pathlib import Path
from functools import wraps
from loguru import logger

def _csv_safe_patcher(record):
    """
    内部冷门黑科技：清洗器（Patcher）
    在日志流入管道前，拦截 message，将其中的换行符和双引号进行标准 CSV 转义，
    防止多行 Traceback 导致生成的 CSV 表格彻底坍塌、错位。
    """
    msg = record["message"]
    # 1. 规避双引号：将 " 替换为 "" (CSV 官方转义规范)
    msg = msg.replace('"', '""')
    # 2. 规避换行符：将换行符替换为特殊的可见符号 \\n，让其保持在单行内
    if "\n" in msg:
        msg = msg.replace("\n", "\\n")

    # 重新注入回记录中（专门创建一个干净的字段供 CSV 消费，不污染控制台）
    record["extra"]["csv_message"] = msg


# 3. 增加这个辅助函数和注册
def on_exit():
    # 注意：在程序异常退出时，logger 依然可以正常记录
    if sys.exc_info()[0]:
        logger.error(f"⚠️ [System] 训练管道发生异常终止: {sys.exc_info()[1]}")
    else:
        logger.info("✅ [System] 训练管道正常结束。")



def log_prepare(model_name: str = "resnet50", dataset_name: str = "TID2013"):
    """
    全局工业日志底座一键配置。
    🚀 已将模型名剔除，完全变更为：年月日_时分秒_任务_数据集 格式
    """
    # 1. 动态计算根目录 (src/plugins/logging_utils.py -> 向上两级是项目根目录)
    UTILS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = UTILS_DIR.parent.parent

    # 2. 强行清除 Loguru 默认全局处理器（拒绝 DEBUG 刷屏卡死 tqdm）
    logger.remove()

    # 3. 动态配置全局清洗器（Patcher），让全线日志自动获得 CSV 安全字段
    logger.configure(patcher=_csv_safe_patcher)

    # 4. 配置【控制台】输出：精简、高清、不打扰 tqdm
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format=console_format
    )

    # 5. 计算落盘路径：results/train_logs/
    log_dir = PROJECT_ROOT / "results" / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 💎【核心修改】：时间格式升级为 年月日_时分秒 (例如：20260519_143025)
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"{current_timestamp}_{model_name.lower()}_{dataset_name.lower()}"

    # 6. 重定向输出流到标准 .log (TXT文本日志) - 留存 DEBUG 全量细节
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <5} | {name}:{function}:{line} - {message}"
    logger.add(
        log_dir / f"{base_filename}.log",
        rotation="500 MB",
        level="DEBUG",
        format=file_format,
        encoding="utf-8"
    )



    # 格式化输出到 CSV 日志
    csv_format = "{time:YYYY-MM-DD HH:mm:ss.SSS},{level},{name},{function},{line},\"{extra[csv_message]}\""
    csv_header = "timestamp,level,module,function,line,message\n"
    csv_path = log_dir / f"{base_filename}.csv"

    # 💎 核心修复 1：精准顺应 Loguru 的底层 rotation 回调签名！
    def csv_rotation_callback(message, file_object):
        """Loguru 轮转新日志文件时自动触发，为其续上标准 CSV 表头"""
        file_object.write(csv_header)

    # 💎 核心修复 2：冷启动铁壁防御。如果文件还不存在（首次运行），用原生 Python 强行注入首行表头
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    # 核心安全通道闭环
    logger.add(
        csv_path,
        rotation=csv_rotation_callback,
        level="DEBUG",
        format=csv_format,
        encoding="utf-8"
    )

    logger.info(f"⚙️  [System] 全局工业日志底座安全初始化成功！")
    logger.info(f"📁 [Outputs] TXT文本日志: results/logs/{base_filename}.log")
    logger.info(f"📁 [Outputs] CSV图表日志: results/logs/{base_filename}.csv")

    atexit.register(on_exit)
    return base_filename



def time_it(func):
    """
    装饰器：精准统计核心训练、评估或数据处理函数的执行耗时。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        module_name = func.__module__.split('.')[-1]
        logger.info(f"⏱️  [Timer] [{module_name}] 执行 {func.__name__} 耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper