# src/data/loaders.py
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import re
from typing import List, Optional


def clean_and_split_line(line: str, possible_delimiters: List[str] = None) -> Optional[List[str]]:
    """
    清洗行内容并智能识别分隔符

    Args:
        line: 原始行字符串
        possible_delimiters: 可能的分隔符列表，默认 [',', '|', ' ', '\t']

    Returns:
        清洗后的字段列表，如果无法识别则返回 None
    """
    if not line or not line.strip():
        return None

    # 1. 过滤各种引号：""、''、“”、‘’
    quote_pattern = r'["“”\'\'‘’]'  # 匹配所有类型的引号
    cleaned = re.sub(quote_pattern, '', line.strip())

    # 2. 默认分隔符
    if possible_delimiters is None:
        possible_delimiters = [',', '|', ' ', '\t']

    # 3. 尝试识别分隔符
    delimiter = None
    for delim in possible_delimiters:
        if delim in cleaned:
            # 避免空格误判（如果只有空格但没有其他分隔符）
            if delim == ' ' and not any(d in cleaned for d in [',', '|', '\t']):
                # 多个连续空格作为一个分隔符
                delimiter = r'\s+'
                break
            elif delim != ' ':
                delimiter = delim
                break

    # 4. 如果没有识别到分隔符，整行作为一个字段
    if delimiter is None:
        return [cleaned]

    # 5. 按分隔符拆分
    if delimiter == r'\s+':
        fields = re.split(r'\s+', cleaned)
    else:
        fields = cleaned.split(delimiter)

    # 6. 清理每个字段的首尾空白
    fields = [f.strip() for f in fields if f.strip()]

    return fields



class BaseMetadataLoader(ABC):
    """所有数据集加载器的抽象基类"""
    @abstractmethod
    def load(self, meta_file: Path) -> pd.DataFrame:
        """必须返回标准化的 DataFrame，包含 sample_id 和 mos 列"""
        pass

    def _ensure_extension(self, df: pd.DataFrame, ext: str) -> pd.DataFrame:
        """确保 sample_id 有指定扩展名"""
        df['sample_id'] = df['sample_id'].apply(
            lambda x: x if x.lower().endswith(ext) else x + ext
        )
        return df

    def _parse_with_cleaner(self, meta_file: Path, delimiter_hint: List[str] = None) -> pd.DataFrame:
        """
        使用 clean_and_split_line 逐行解析文件
        适用于格式不标准的元数据文件
        """
        records = []
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                fields = clean_and_split_line(line, delimiter_hint)
                if not fields:
                    continue

                # 假设前两列是 sample_id 和 mos
                if len(fields) >= 2:
                    records.append({
                        'sample_id': fields[0],
                        'mos': fields[1]
                    })
                else:
                    logger.warning(f"行 {line_num} 字段数不足: {fields}")

        if not records:
            raise ValueError(f"未从 {meta_file} 解析到有效数据")

        return pd.DataFrame(records)




class Tid2013Loader(BaseMetadataLoader):
    def load(self, meta_file: Path) -> pd.DataFrame:
        # 使用 sep=r'\s+' 处理空格分隔，header=None 因为没有表头
        # names 明确指定列顺序：第一列是 MOS，第二列是 ID
        try:
            df = pd.read_csv(
                meta_file,
                sep=r'\s+',
                header=None,
                names=['mos', 'sample_id']
            )
            df = self._ensure_extension(df, '.bmp')
            return df[['sample_id', 'mos']]

        except Exception as e:
            # 如果 pandas 读取失败，使用清洗函数兜底
            logger.warning(f"TID2013 pandas 读取失败，尝试逐行解析: {e}")
            df = self._parse_with_cleaner(meta_file, delimiter_hint=[r'\s+'])
            df = self._ensure_extension(df, '.bmp')
            return df[['sample_id', 'mos']]



class KonvidLoader(BaseMetadataLoader):
    def load(self, meta_file: Path) -> pd.DataFrame:
        # Konvid 专用逻辑
        try:
            df = pd.read_csv(
                meta_file,
                quotechar='"',
                skipinitialspace=True
            )
            df = df.rename(columns={"flickr_id": "sample_id", "mos": "mos"})
            df['sample_id'] = df['sample_id'].astype(str).str.replace(r'["\s]', '', regex=True)
            df = self._ensure_extension(df, '.mp4')
            return df[['sample_id', 'mos']]

        except Exception as e:
            logger.warning(f"Konvid-1k pandas 读取失败，尝试逐行解析: {e}")
            # Konvid 通常是逗号分隔
            df = self._parse_with_cleaner(meta_file, delimiter_hint=[','])
            df = self._ensure_extension(df, '.mp4')
            return df[['sample_id', 'mos']]



class T2VqaLoader(BaseMetadataLoader):
    def load(self, meta_file: Path) -> pd.DataFrame:
        # T2VQA 专用逻辑
        try:
            df = pd.read_csv(
                meta_file,
                sep='|',
                header=None,
                names=['sample_id', 'description', 'mos']
            )
            df['description'] = df['description'].str.strip()
            df = self._ensure_extension(df, '.mp4')
            return df[['sample_id', 'mos']]
            # TODO:
            # return df[['sample_id', 'description', 'mos']]

        except Exception as e:
            logger.warning(f"T2VQA pandas 读取失败，尝试逐行解析: {e}")
            # T2VQA 通常是竖线分隔
            df = self._parse_with_cleaner(meta_file, delimiter_hint=['|'])
            # 注意：使用清洗函数时没有 description 列
            df = self._ensure_extension(df, '.mp4')
            return df[['sample_id', 'mos']]