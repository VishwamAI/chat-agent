from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from evaluation.abstract_evaluator import AbstractEvaluator
from evaluation.metrics.abstract_metric import AbstractMetric
from evaluation.metrics.squad_metric import SquadMetric
from evaluation.metrics.bleu_metric import BleuMetric
from evaluation.metrics.chrf_metric import ChrfMetric
from evaluation.metrics.meteor_metric import MeteorMetric
from evaluation.metrics.rouge_metric import RougeMetric
from evaluation.metrics.bertscore_metric import BertscoreMetric
from evaluation.metrics.squad_bertscore_metric import SquadBertscoreMetric
from evaluation.metrics.squad_bleu_metric import SquadBleuMetric
from evaluation.metrics.squad_chrf_metric import SquadChrfMetric
from evaluation.metrics.squad_meteor_metric import SquadMeteorMetric
from evaluation.metrics.squad_rouge_metric import SquadRougeMetric
from evaluation.metrics.squad_f1_metric import SquadF1Metric
from evaluation.metrics.squad_exact_match_metric import SquadExactMatchMetric
from evaluation.metrics.squad_squad_metric import SquadSquadMetric
from evaluation.metrics.squad_squad_bertscore_metric import SquadSquadBertscoreMetric
from evaluation.metrics.squad_squad_bleu_metric import SquadSquadBleuMetric
from evaluation.metrics.squad_squad_chrf_metric import SquadSquadChrfMetric
from evaluation.metrics.squad_squad_meteor_metric import SquadSquadMeteorMetric
from evaluation.metrics.squad_squad_rouge_metric import SquadSquadRougeMetric
from evaluation.metrics.squad_squad_f1_metric import SquadSquadF1Metric
from evaluation.metrics.squad_squad_exact_match_metric import SquadSquadExactMatchMetric

class Evaluator(AbstractEvaluator):
    def __init__(self, task_name: str, metrics: Optional[List[str]] = None):
        super().__init__(task_name)

        self.metrics = metrics or []

        self.metric_dict = {
            "squad": {
                "squad": SquadMetric(),
                "bertscore": SquadBertscoreMetric(),
                "bleu": SquadBleuMetric(),
                "chrf": SquadChrfMetric(),
                "meteor": SquadMeteorMetric(),
                "rouge": SquadRougeMetric(),
                "f1": SquadF1Metric(),
                "exact_match": SquadExactMatchMetric(),
            },
            "squad_squad": {
                "squad": SquadSquadMetric(),
                "bertscore": SquadSquadBertscoreMetric(),
                "bleu": SquadSquadBleuMetric(),
                "chrf": SquadSquadChrfMetric(),
                "meteor": SquadSquadMeteorMetric(),
                "rouge": SquadSquadRougeMetric(),
                "f1": SquadSquadF1Metric(),
                "exact_match": SquadSquadExactMatchMetric(),
            },
        }

        self.task_metric_dict = self.metric_dict.get(task_name, {})

        for metric_name in self.metrics:
            if metric_name not in self.task_metric_dict:
                raise ValueError(f"Invalid metric name: {metric_name}")

    def evaluate(self, predictions: List[Dict[str, str]], references: List[Dict[str, str]]) -> Dict[str, float]:
        scores = defaultdict(list)

        for prediction, reference in zip(predictions, references):
            for metric_name, metric in self.task_metric_dict.items():
                if metric_name in self.metrics:
                    score = metric(prediction, reference)
                    scores[metric_name].append(score)

        results = {}
        for metric_name, score_list in scores.items():
            results[metric_name] = sum(score_list) / len(score_list)

        return results
