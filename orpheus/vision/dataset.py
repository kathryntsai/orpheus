from torch.utils.data import Dataset, DataLoader
import torch
from typing import List
import os
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from utils.utils import validate_dataframe


class EmbeddingDatasetCoxRegression(Dataset):
    """
    Serves embeddings from .pt files
    emb_file_paths: List[str] = list of paths to pt files
    case_ids: List[str] = list of case ids
    scores: List[float] = list of scores
    """

    def __init__(
        self,
        df,
    ) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return {
            "x": torch.load(row["input_visual_embedding_path"]).float(),
            "tte": torch.tensor(row["tte"]).float(),
            "censor": torch.tensor(row['censor']).int(),
            "case_id": row["case_id"],
            "output_visual_embedding_path": row["output_visual_embedding_path"],
            "split": row["split"],
        }
    


class EmbeddingDatasetLogisticRegression(Dataset):
    """
    Serves embeddings from .pt files
    emb_file_paths: List[str] = list of paths to pt files
    case_ids: List[str] = list of case ids
    scores: List[float] = list of scores
    """

    def __init__(
        self,
        df,
    ) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return {
            "x": torch.load(row["input_visual_embedding_path"]).float(),
            "y": torch.tensor(row["class"]).int(),
            "case_id": row["case_id"],
            "output_visual_embedding_path": row["output_visual_embedding_path"],
            "split": row["split"],
        }
    

class EmbeddingDataset(Dataset):
    """
    Serves embeddings from .pt files
    emb_file_paths: List[str] = list of paths to pt files
    case_ids: List[str] = list of case ids
    scores: List[float] = list of scores
    """

    def __init__(
        self,
        df,
    ) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return {
            "x": torch.load(row["input_visual_embedding_path"]).float(),
            "y": torch.tensor(row["score"]).float(),
            "case_id": row["case_id"],
            "output_visual_embedding_path": row["output_visual_embedding_path"],
            "split": row["split"],
        }
    


class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataframe_path: str,
        num_workers: int = 0,
        batch_size: int = 1,
        task: str = "regression",   # "regression" | "binary" | "survival"
    ):
        super().__init__()
        assert task in {"regression", "binary", "survival"}, "Invalid data.task"
        self.dataframe_path = dataframe_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.task = task
        self.save_hyperparameters()

        self.df = pd.read_csv(dataframe_path, low_memory=False)
        self.df = self.df[self.df["input_visual_embedding_path"] != "NONE"]
        validate_dataframe(self.df)

        # Placeholders populated in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.predict_ds = None

    def _make_dataset(self, df_split: pd.DataFrame):
        if self.task == "regression":
            return EmbeddingDataset(df_split)
        elif self.task == "binary":
            return EmbeddingDatasetLogisticRegression(df_split)
        else:
            return EmbeddingDatasetCoxRegression(df_split)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            assert "train" in self.df["split"].unique(), "No train split in dataframe"
            assert "val"   in self.df["split"].unique(), "No val split in dataframe"
            self.train_ds = self._make_dataset(self.df.loc[self.df["split"] == "train"])
            self.val_ds   = self._make_dataset(self.df.loc[self.df["split"] == "val"])

        elif stage == "test":
            assert "test" in self.df["split"].unique(), "No test split in dataframe"
            self.test_ds = self._make_dataset(self.df.loc[self.df["split"] == "test"])

        elif stage == "predict":
            self.predict_ds = self._make_dataset(self.df)

        else:
            raise NotImplementedError(f"Unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,  # flip to True to shuffle during training but this keeps everytihng standard for now
            sampler=None,
            batch_sampler=None,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
    

if __name__ == "__main__":
    pass
