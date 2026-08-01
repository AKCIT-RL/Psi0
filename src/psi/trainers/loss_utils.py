import torch


def apply_action_dim_weights(
    elementwise_loss: torch.Tensor,
    hand_loss_weight: float,
    hand_action_dims: list[int],
) -> torch.Tensor:
    if hand_loss_weight == 1.0:
        return elementwise_loss

    weights = torch.ones(
        elementwise_loss.shape[-1],
        dtype=elementwise_loss.dtype,
        device=elementwise_loss.device,
    )
    weights[hand_action_dims] = hand_loss_weight
    weights = weights / weights.mean()
    return elementwise_loss * weights