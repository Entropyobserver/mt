from typing import Dict, List, Optional
import evaluate
import numpy as np


class BaseEvaluator:
    def __init__(
        self,
        use_comet: bool = False,
        comet_model: str = "Unbabel/wmt22-comet-da",
    ):
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")
        self.use_comet = use_comet
        self.comet_model = None

        if self.use_comet:
            try:
                from comet import download_model, load_from_checkpoint
                self.comet_model = load_from_checkpoint(download_model(comet_model))
            except Exception as e:
                print(f"COMET loading failed: {e}")
                self.use_comet = False

    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
        )
        precisions = result["precisions"]
        return {
            "bleu":   result["bleu"],
            "bleu_1": precisions[0] if precisions else 0.0,
            "bleu_2": precisions[1] if len(precisions) > 1 else 0.0,
            "bleu_3": precisions[2] if len(precisions) > 2 else 0.0,
            "bleu_4": precisions[3] if len(precisions) > 3 else 0.0,
        }

    def compute_chrf(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.chrf.compute(predictions=predictions, references=references)
        return {"chrf": result["score"]}

    def compute_comet(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Optional[float]]:
        if not self.use_comet or self.comet_model is None:
            return {"comet": None, "comet_std": None}

        comet_data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        try:
            result = self.comet_model.predict(
                comet_data,
                batch_size=4,
                accelerator="auto",
            )
            return {
                "comet":     float(result["system_score"]),
                "comet_std": float(np.std(result["scores"])),
            }
        except Exception as e:
            print(f"COMET prediction failed: {e}")
            return {"comet": None, "comet_std": None}

    def evaluate_all(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        if self.use_comet and len(sources) != len(predictions):
            raise ValueError("sources, predictions, and references must have the same length")

        sources     = [s.strip() for s in sources]
        predictions = [p.strip() for p in predictions]
        references  = [r.strip() for r in references]

        metrics = {}
        metrics.update(self.compute_bleu(predictions, references))
        metrics.update(self.compute_chrf(predictions, references))
        if self.use_comet and sources:
            metrics.update(self.compute_comet(sources, predictions, references))

        return metrics