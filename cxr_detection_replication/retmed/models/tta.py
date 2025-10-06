import torch
from mmdet.models.test_time_augs.det_tta import DetTTAModel
from mmdet.structures import DetDataSample
from typing import List, Optional


class CustomDetTTAModel(DetTTAModel):
    """Custom TTA model that handles data type consistency issues."""
    
    def _merge_single_sample(self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions from multiple augmented samples.
        
        Args:
            data_samples (List[DetDataSample]): List of data samples from
                different augmentations.
                
        Returns:
            DetDataSample: Merged data sample.
        """
        if len(data_samples) == 1:
            return data_samples[0]
        
        # Get the first sample as base
        merged_data_sample = data_samples[0].clone()
        
        # Collect predictions from all augmentations
        aug_pred_instances = []
        for data_sample in data_samples:
            if hasattr(data_sample, 'pred_instances_3d'):
                aug_pred_instances.append(data_sample.pred_instances_3d)
            else:
                aug_pred_instances.append(data_sample.pred_instances)
        
        # Merge predictions
        merged_pred_instances = self._merge_pred_instances(aug_pred_instances)
        
        # Update the merged data sample
        if hasattr(merged_data_sample, 'pred_instances_3d'):
            merged_data_sample.pred_instances_3d = merged_pred_instances
        else:
            merged_data_sample.pred_instances = merged_pred_instances
            
        return merged_data_sample
    
    def _merge_pred_instances(self, pred_instances_list: List) -> object:
        """Merge prediction instances from multiple augmentations.
        
        Args:
            pred_instances_list (List): List of prediction instances.
            
        Returns:
            object: Merged prediction instances.
        """
        if len(pred_instances_list) == 1:
            return pred_instances_list[0]
        
        # Get the first instance as base
        merged_instances = pred_instances_list[0].clone()
        
        # Collect all predictions
        all_bboxes = []
        all_scores = []
        all_labels = []
        
        for pred_instances in pred_instances_list:
            if hasattr(pred_instances, 'bboxes') and pred_instances.bboxes is not None:
                all_bboxes.append(pred_instances.bboxes)
            if hasattr(pred_instances, 'scores') and pred_instances.scores is not None:
                all_scores.append(pred_instances.scores)
            if hasattr(pred_instances, 'labels') and pred_instances.labels is not None:
                # Ensure labels are Long type
                labels = pred_instances.labels
                if labels.dtype != torch.long:
                    labels = labels.long()
                all_labels.append(labels)
        
        # Concatenate predictions
        if all_bboxes:
            merged_instances.bboxes = torch.cat(all_bboxes, dim=0)
        if all_scores:
            merged_instances.scores = torch.cat(all_scores, dim=0)
        if all_labels:
            # Ensure all labels are Long type before concatenation
            all_labels = [labels.long() if labels.dtype != torch.long else labels 
                         for labels in all_labels]
            merged_instances.labels = torch.cat(all_labels, dim=0)
        
        return merged_instances 