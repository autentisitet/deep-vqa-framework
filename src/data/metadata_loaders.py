# src/data/metadata_loaders.py
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


def clean_and_split_line(line: str, possible_delimiters: List[str] = None) -> Optional[List[str]]:
    """
    Clean line content and intelligently identify delimiters.

    Args:
        line: Raw line string
        possible_delimiters: List of possible delimiters, defaults to [',', '|', ' ', '\t']

    Returns:
        List of cleaned fields, or None if no delimiter can be identified
    """
    if not line or not line.strip():
        return None

    # Remove various quotes: "", '', “”, ‘’
    quote_pattern = r'["“”\'\'‘’]'  # Matches all types of quotes
    cleaned = re.sub(quote_pattern, "", line.strip())

    if possible_delimiters is None:
        possible_delimiters = [",", "|", " ", "\t"]

    # Identify delimiter
    delimiter = None
    for delim in possible_delimiters:
        if delim in cleaned:
            # Avoid misinterpreting spaces: if there are only spaces but no other delimiters,
            # treat multiple consecutive spaces as the separator
            if delim == " " and not any(d in cleaned for d in [",", "|", "\t"]):
                delimiter = r"\s+"
                break
            elif delim != " ":
                delimiter = delim
                break

    # If no delimiter is detected, treat the entire line as a single field.
    if delimiter is None:
        return [cleaned]

    # Split by delimiter
    if delimiter == r"\s+":
        fields = re.split(r"\s+", cleaned)
    else:
        fields = cleaned.split(delimiter)

    # Strip whitespace from each field and filter out empty strings
    fields = [f.strip() for f in fields if f.strip()]

    return fields




class BaseMetadataLoader(ABC):
    """Abstract base class for all dataset metadata loaders."""

    @abstractmethod
    def load(self, meta_file: Path) -> pd.DataFrame:
        """
        Load metadata from file.

        Returns:
            DataFrame with at least 'sample_id' and 'mos' columns.
        """
        pass


    def _ensure_extension(self, df: pd.DataFrame, ext: str) -> pd.DataFrame:
        """Make sure sample_id has a specified file extension."""
        df["sample_id"] = df["sample_id"].apply(
            lambda x: x if x.lower().endswith(ext) else x + ext
        )
        return df


    def _parse_with_cleaner(self, meta_file: Path, delimiter_hint: List[str] = None) -> pd.DataFrame:
        """
        Parse file line by line using clean_and_split_line.

        Useful for metadata files with non-standard formats.
        """
        records = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                fields = clean_and_split_line(line, delimiter_hint)
                if not fields:
                    continue

                # 假设前两列是 sample_id 和 mos
                if len(fields) >= 2:
                    records.append({"sample_id": fields[0], "mos": fields[1]})
                else:
                    logger.warning(f"Line {line_num} has insufficient fields: {fields}")

        if not records:
            raise ValueError(f"No valid data parsed from {meta_file}")

        return pd.DataFrame(records)




class Tid2013Loader(BaseMetadataLoader):
    """Loader for TID2013 dataset."""

    def _add_reference_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df["reference_id"] = (
            df["sample_id"]
            .astype(str)
            .str.extract(r"^([iI]\d{1,3})_", expand=False)
            .str.lower()
        )
        df["reference_id"] = df["reference_id"].fillna(
            df["sample_id"].astype(str).str.replace(r"\.[^.]+$", "", regex=True).str.lower()
        )
        return df


    def load(self, meta_file: Path) -> pd.DataFrame:
        # 使用 sep=r'\s+' 处理空格分隔，header=None 因为没有表头
        # names 明确指定列顺序：第一列是 MOS，第二列是 ID
        try:
            df = pd.read_csv(meta_file, sep=r"\s+", header=None, names=["mos", "sample_id"])
            df = self._ensure_extension(df, ".bmp")
            df = self._add_reference_id(df)
            return df[["sample_id", "mos", "reference_id"]]

        except Exception as e:
            # If pandas read fails, use a cleanup function as a fallback.
            logger.warning(f"TID2013 pandas read failed, attempting to parse line by line: {e}")
            records = []
            with open(meta_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    fields = clean_and_split_line(line, delimiter_hint=[" ", "\t"])
                    if not fields:
                        continue
                    if len(fields) < 2:
                        logger.warning(f"TID2013 line {line_num} has insufficient fields: {fields}")
                        continue
                    records.append({"mos": fields[0], "sample_id": fields[1]})
            if not records:
                raise ValueError(f"No valid TID2013 metadata parsed from {meta_file}")
            df = pd.DataFrame(records)
            df = self._ensure_extension(df, ".bmp")
            df["mos"] = pd.to_numeric(df["mos"], errors="raise")
            df = self._add_reference_id(df)
            return df[["sample_id", "mos", "reference_id"]]




class KonvidLoader(BaseMetadataLoader):
    """Loader for KoNViD-1k dataset."""

    def load(self, meta_file: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(meta_file, quotechar='"', skipinitialspace=True)
            df = df.rename(columns={"flickr_id": "sample_id", "mos": "mos"})
            df["sample_id"] = df["sample_id"].astype(str).str.replace(r'["\s]', "", regex=True)
            df = self._ensure_extension(df, ".mp4")
            return df[["sample_id", "mos"]]

        except Exception as e:
            logger.warning(f"Konvid-1k pandas read failed, attempting to parse line by line: {e}")
            records = []
            with open(meta_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    fields = clean_and_split_line(line, delimiter_hint=[","])
                    if not fields:
                        continue
                    if line_num == 1 and fields[0].lower() == "flickr_id":
                        continue
                    if len(fields) < 2:
                        logger.warning(f"KoNViD-1k line {line_num} has insufficient fields: {fields}")
                        continue
                    records.append({"sample_id": fields[0], "mos": fields[1]})
            if not records:
                raise ValueError(f"No valid KoNViD-1k metadata parsed from {meta_file}")
            df = pd.DataFrame(records)
            df = self._ensure_extension(df, ".mp4")
            df["mos"] = pd.to_numeric(df["mos"], errors="raise")
            return df[["sample_id", "mos"]]




class T2VqaLoader(BaseMetadataLoader):
    """Loader for T2VQA-DB dataset."""

    def load(self, meta_file: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                meta_file,
                sep="|",
                header=None,
                names=["sample_id", "description", "mos"],
            )
            df["description"] = df["description"].str.strip()
            df = self._ensure_extension(df, ".mp4")
            # Return description column if needed:
            # return df[["sample_id", "description", "mos"]]
            return df[["sample_id", "mos"]]

        except Exception as e:
            logger.warning(f"T2VQA pandas read failed, attempting to parse line by line: {e}")
            records = []
            with open(meta_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    fields = clean_and_split_line(line, delimiter_hint=["|"])
                    if not fields:
                        continue
                    if len(fields) < 3:
                        logger.warning(f"T2VQA line {line_num} has insufficient fields: {fields}")
                        continue
                    records.append({"sample_id": fields[0], "mos": fields[2]})
            if not records:
                raise ValueError(f"No valid T2VQA metadata parsed from {meta_file}")
            df = pd.DataFrame(records)
            df = self._ensure_extension(df, ".mp4")
            df["mos"] = pd.to_numeric(df["mos"], errors="raise")
            return df[["sample_id", "mos"]]
