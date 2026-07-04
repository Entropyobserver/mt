import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseTrainer(ABC):
    def __init__(self, model_name: str, src_lang: str, tgt_lang: str):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")

    @abstractmethod
    def setup_model(self, **kwargs):
        pass

    def tokenize(self, examples: Dict) -> Dict:
        max_length = getattr(self, "max_length", 128)

        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        self.tokenizer.tgt_lang = self.tgt_lang
        labels = self.tokenizer(
            text_target=examples["target"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        # Ignore padding tokens in the loss.
        model_inputs["labels"] = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        train_dataset = train_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        return train_dataset, val_dataset

    def training_args(self, config: Dict) -> Seq2SeqTrainingArguments:
        eval_steps = config.get("eval_steps", 200)
        save_steps = config.get("save_steps", eval_steps)

        # HuggingFace requires save_steps to be a multiple of eval_steps.
        if save_steps % eval_steps != 0:
            save_steps = eval_steps

        return Seq2SeqTrainingArguments(
            output_dir=config["output_dir"],
            seed=config.get("seed", 42),
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            per_device_eval_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_steps=config.get("warmup_steps", 100),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            fp16=config.get("fp16", True),
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            eval_strategy="steps",
            eval_steps=eval_steps,
            metric_for_best_model=config.get("metric_for_best_model", "bleu"),
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=config.get("generation_max_length", config.get("max_length", 128)),
            generation_num_beams=config.get("generation_num_beams", 5),
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=config.get("save_total_limit", 2),
            load_best_model_at_end=True,
            logging_steps=config.get("logging_steps", 50),
            remove_unused_columns=False,
            report_to=[],
        )

    def train(self, train_data: List[Dict], val_data: List[Dict], config: Dict) -> Dict:
        output_dir = Path(config["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        self.max_length = config.get("max_length", 128)
        model = self.setup_model(**config)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        model.config.forced_bos_token_id = forced_bos_token_id
        model.generation_config.forced_bos_token_id = forced_bos_token_id
        train_dataset, val_dataset = self.prepare_datasets(train_data, val_data)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model,
            padding=True,
        )

        callbacks = []
        if config.get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config["early_stopping_patience"],
                    early_stopping_threshold=config.get("early_stopping_threshold", 0.001),
                )
            )

        trainer = Seq2SeqTrainer(
            model=model,
            args=self.training_args(config),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        return {
            "model": model,
            "trainer": trainer,
            "bleu": eval_result.get("eval_bleu", 0.0),
            "chrf": eval_result.get("eval_chrf", 0.0),
            "loss": eval_result.get("eval_loss", 0.0),
        }

    def compute_metrics(self, eval_pred) -> Dict:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = self.bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )
        chrf = self.chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {"bleu": bleu["bleu"], "chrf": chrf["score"]}

    def generate_predictions(
        self,
        model,
        dataset,
        batch_size: int = 8,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> List[str]:
        predictions = []
        device = next(model.parameters()).device
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        for i in range(0, len(dataset), batch_size):
            batch = dataset.samples[i : i + batch_size]
            sources = [sample.source for sample in batch]

            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(
                sources,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    forced_bos_token_id=forced_bos_token_id,
                )

            predictions.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

        return predictions
