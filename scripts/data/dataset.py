import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch.utils.data import Dataset


@dataclass
class TranslationSample:
    source: str
    target: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "metadata": self.metadata or {},
        }


class TranslationDataset(Dataset):
    def __init__(
        self,
        samples: List[TranslationSample],
        src_lang: str = "eng_Latn",
        tgt_lang: str = "nob_Latn",
    ):
        self.samples = samples
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx].to_dict()

    @classmethod
    def from_json(
        cls,
        path: Path,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "nob_Latn",
    ) -> "TranslationDataset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        samples = [
            TranslationSample(
                source=item["source"],
                target=item["target"],
                metadata=item.get("metadata"),
            )
            for item in data
        ]

        return cls(samples, src_lang=src_lang, tgt_lang=tgt_lang)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "nob_Latn",
    ) -> "TranslationDataset":
        samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                samples.append(
                    TranslationSample(
                        source=item["source"],
                        target=item["target"],
                        metadata=item.get("metadata"),
                    )
                )

        return cls(samples, src_lang=src_lang, tgt_lang=tgt_lang)

    @classmethod
    def from_file(
        cls,
        path: Path,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "nob_Latn",
    ) -> "TranslationDataset":
        suffix = path.suffix.lower()

        if suffix == ".json":
            return cls.from_json(path, src_lang=src_lang, tgt_lang=tgt_lang)

        if suffix == ".jsonl":
            return cls.from_jsonl(path, src_lang=src_lang, tgt_lang=tgt_lang)

        raise ValueError(f"Unsupported dataset file format: {path}")

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [sample.to_dict() for sample in self.samples],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def to_jsonl(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(
                    json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
                )

    def subset(self, size: int, seed: int = 42) -> "TranslationDataset":
        if size >= len(self.samples):
            return self

        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(self.samples)), size))

        return TranslationDataset(
            [self.samples[i] for i in indices],
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )

    def get_statistics(self) -> Dict:
        if not self.samples:
            return {
                "total": 0,
                "avg_source_length": 0,
                "avg_target_length": 0,
                "max_source_length": 0,
                "max_target_length": 0,
            }

        src_lengths = [len(sample.source.split()) for sample in self.samples]
        tgt_lengths = [len(sample.target.split()) for sample in self.samples]

        return {
            "total": len(self.samples),
            "avg_source_length": sum(src_lengths) / len(src_lengths),
            "avg_target_length": sum(tgt_lengths) / len(tgt_lengths),
            "max_source_length": max(src_lengths),
            "max_target_length": max(tgt_lengths),
        }