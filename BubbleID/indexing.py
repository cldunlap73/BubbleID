import numpy as np
import torch


def _tensor_indices_to_numpy(indices):
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy().astype(np.int64)
    return np.asarray(indices, dtype=np.int64)


def _numpy_indices_to_tensor(indices, device):
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def filter_segmentation_outputs(masks, scores, class_values, threshold):
    keep_indices = torch.nonzero(scores > threshold, as_tuple=False).flatten()
    masks = torch.index_select(masks, 0, keep_indices)

    class_values = np.asarray(class_values)
    kept_classes = class_values[_tensor_indices_to_numpy(keep_indices)]

    base_indices = np.where(kept_classes == 0)[0]
    base_masks = masks[_numpy_indices_to_tensor(base_indices, masks.device)]

    return masks, kept_classes, base_masks
