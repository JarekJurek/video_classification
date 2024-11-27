import torch


def aggregate_predictions(predictions):
    """
    Aggregates predictions from multiple frames.
    Args:
        predictions: Tensor of shape [num_frames, num_classes]
    Returns:
        Aggregated prediction: Tensor of shape [num_classes]
    """
    return torch.mean(predictions, dim=0)
