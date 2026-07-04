from pathlib import Path

from scripts.data.dataset import TranslationDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataManager:
    """
    Loads train/val/test splits from JSON or JSONL files.
    Config is a plain dict loaded from config.yaml.
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = PROJECT_ROOT / config["paths"]["data_dir"]

    def load_splits(self, train_subset_size=None, reverse=False):
        train_path = self.data_dir / self.config["data"]["train"]
        val_path = self.data_dir / self.config["data"]["val"]
        test_path = self.data_dir / self.config["data"]["test"]

        src_lang = self.config["model"]["src_lang"]
        tgt_lang = self.config["model"]["tgt_lang"]

        if reverse:
            src_lang, tgt_lang = tgt_lang, src_lang

        train_ds = TranslationDataset.from_file(train_path, src_lang=src_lang, tgt_lang=tgt_lang)
        val_ds = TranslationDataset.from_file(val_path, src_lang=src_lang, tgt_lang=tgt_lang)
        test_ds = TranslationDataset.from_file(test_path, src_lang=src_lang, tgt_lang=tgt_lang)

        if reverse:
            for ds in (train_ds, val_ds, test_ds):
                for sample in ds.samples:
                    sample.source, sample.target = sample.target, sample.source

        if train_subset_size is not None and train_subset_size < len(train_ds):
            train_ds = train_ds.subset(train_subset_size, seed=self.config["project"]["seed"])

        return train_ds, val_ds, test_ds
