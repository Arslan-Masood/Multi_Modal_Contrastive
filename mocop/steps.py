from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

            
def _validation_epoch_end(self, validation_step_outputs, is_regression=False):
    # Handle single dictionary case
    if isinstance(validation_step_outputs, dict):
        validation_step_outputs = [validation_step_outputs]
    
    # Handle empty case
    if not validation_step_outputs:
        return
        
    # Handle list of dictionaries case
    if not isinstance(validation_step_outputs[0], list):
        validation_step_outputs = [validation_step_outputs]

    for i, validation_step_output in enumerate(validation_step_outputs):
        # If single dictionary, wrap it in a list
        if isinstance(validation_step_output, dict):
            validation_step_output = [validation_step_output]
            
        all_logs = [o["log"] for o in validation_step_output]
        all_logs = {k: [log[k] for log in all_logs] for k in all_logs[0].keys()}
        all_logs = {k: np.mean(v) for k, v in all_logs.items()}

        if (
            "outputs" in validation_step_output[0]
            and "labels" in validation_step_output[0]
        ):
            all_outputs = torch.cat([o["outputs"] for o in validation_step_output])
            all_labels = torch.cat([o["labels"] for o in validation_step_output])
            metric_func = _supervised_metric
            if is_regression:
                metric_func = _supervised_metric_regression
            metrics = metric_func(all_labels, all_outputs)
            metrics = {f"val/{k}": v for k, v in metrics.items()}
            all_logs.update(metrics)

        if i != 0:
            all_logs = {f"{k}_{i}": v for k, v in all_logs.items()}
        print(all_logs)
        
        # Log metrics only if not in sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(all_logs, on_step=False, on_epoch=True)

def _validation_epoch_end_all_cell_lines(self, validation_step_outputs, is_regression=False):
    # Handle single dictionary case
    if isinstance(validation_step_outputs, dict):
        validation_step_outputs = [validation_step_outputs]
    
    # Handle empty case
    if not validation_step_outputs:
        return
        
    # Handle list of dictionaries case
    if not isinstance(validation_step_outputs[0], list):
        validation_step_outputs = [validation_step_outputs]

    for i, validation_step_output in enumerate(validation_step_outputs):
        # If single dictionary, wrap it in a list
        if isinstance(validation_step_output, dict):
            validation_step_output = [validation_step_output]
            
        # Extract logs and calculate means
        all_logs = [o["log"] for o in validation_step_output]
        
        # List of metrics we want to extract
        desired_metrics = [
            'val/loss_ab',
            'val/loss_ba', 
            'val/morphological_loss',
            'val/genomic_loss',
            'val/n_cell_lines',
            'val/loss',
            'val/n_valid_losses'
        ]
        
        # Filter and calculate means only for desired metrics
        filtered_logs = {}
        for metric in desired_metrics:
            if metric in all_logs[0]:
                filtered_logs[metric] = np.mean([log[metric] for log in all_logs])

        if i != 0:
            filtered_logs = {f"{k}_{i}": v for k, v in filtered_logs.items()}
        print(filtered_logs)
        
        # Log metrics only if not in sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(filtered_logs, on_step=False, on_epoch=True)
    
def _supervised_metric(supervised_labels, supervised_outputs):
    metric = defaultdict(list)
    for labels, logits in zip(supervised_labels.T, supervised_outputs.T):
        mask = labels != -1
        masked_labels = torch.masked_select(labels, mask).cpu().detach().numpy()
        if len(masked_labels) == 0 or np.max(masked_labels) == np.min(masked_labels):
            metric["auroc"].append(0.5)
            metric["auprc"].append(0.5)
            continue

        masked_logits = torch.masked_select(logits, mask).cpu().detach().numpy()
        metric["auroc"].append(roc_auc_score(masked_labels, masked_logits))
        metric["auprc"].append(average_precision_score(masked_labels, masked_logits))

    metric_single_task = {
        f"{k}_{i}": v_ for k, v in metric.items() for i, v_ in enumerate(v)
    }
    metric_mean = {f"{k}_mean": np.mean(v) for k, v in metric.items()}
    metric_mean.update(metric_single_task)
    return metric_mean


def _supervised_metric_regression(supervised_labels, supervised_outputs):
    metric = defaultdict(list)
    for labels, logits in zip(supervised_labels.T, supervised_outputs.T):
        labels = labels.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()
        metric["mae"].append(mean_absolute_error(labels, logits))
        metric["mse"].append(mean_squared_error(labels, logits))
    metric = {k: np.mean(v) for k, v in metric.items()}
    return metric
