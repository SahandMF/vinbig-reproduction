import torch
from mmdet.models.test_time_augs.det_tta import DetTTAModel
from typing import List


def patch_tta_model():
    """Patch the DetTTAModel to handle data type consistency issues."""
    
    # Store the original method
    original_merge_single_sample = DetTTAModel._merge_single_sample
    
    def _merge_single_sample_patched(self, data_samples: List) -> object:
        """Patched version that ensures consistent data types before calling original method."""
        if len(data_samples) == 1:
            return data_samples[0]
        
        # Fix data types in the predictions before merging
        for data_sample in data_samples:
            if hasattr(data_sample, 'pred_instances') and data_sample.pred_instances is not None:
                if hasattr(data_sample.pred_instances, 'labels') and data_sample.pred_instances.labels is not None:
                    # Ensure labels are Long type
                    if data_sample.pred_instances.labels.dtype != torch.long:
                        data_sample.pred_instances.labels = data_sample.pred_instances.labels.long()
            if hasattr(data_sample, 'pred_instances_3d') and data_sample.pred_instances_3d is not None:
                if hasattr(data_sample.pred_instances_3d, 'labels') and data_sample.pred_instances_3d.labels is not None:
                    # Ensure labels are Long type
                    if data_sample.pred_instances_3d.labels.dtype != torch.long:
                        data_sample.pred_instances_3d.labels = data_sample.pred_instances_3d.labels.long()
        
        # Call the original method with fixed data types
        return original_merge_single_sample(self, data_samples)
    
    # Apply the patch
    DetTTAModel._merge_single_sample = _merge_single_sample_patched