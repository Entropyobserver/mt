import gc
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM

from scripts.model.base_trainer import BaseTrainer


class FullTrainer(BaseTrainer):

    def __init__(self, model_name: str, src_lang: str, tgt_lang: str):
        super().__init__(model_name, src_lang, tgt_lang)

    def setup_model(self, **kwargs):
        # fp32 for stability — full fine-tuning with fp16 can diverge
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )

        # all 600M parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        return model

    def train(self, train_data: List[Dict], val_data: List[Dict], config: Dict) -> Dict:

        # force fp32 regardless of what config says
        config = dict(config)
        config["fp16"] = False

        result = super().train(train_data, val_data, config)

        # full_ft model is large (~2.4GB), delete after evaluation to save disk space
        if config.get("save_final_model", False):
            final_path = Path(config["output_dir"]) / "final_model"
            final_path.mkdir(parents=True, exist_ok=True)
            result["trainer"].save_model(str(final_path))
            result["final_model_path"] = str(final_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result