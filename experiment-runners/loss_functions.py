import torch

def compute_masked_loss(logits, labels, loss_mask):
    """
    Compute causal language modeling (CLM) loss while excluding specific tokens
    (e.g., text prompt tokens) from the loss calculation using a loss mask.

    Args:
        logits (torch.Tensor): Model logits of shape [batch_size, seq_len, vocab_size].
        labels (torch.Tensor): Ground truth token IDs of shape [batch_size, seq_len].
        loss_mask (torch.Tensor): Mask of shape [batch_size, seq_len] where:
                                  1 indicates tokens to include in the loss,
                                  0 excludes tokens (e.g., text prompt tokens).

    Returns:
        torch.Tensor: The average loss for the batch.
    """
    # Shift logits and labels for causal language modeling
    shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()      # [batch_size, seq_len-1]

    # Shift the loss mask to align with shifted labels
    loss_mask = loss_mask[..., 1:].contiguous()      # [batch_size, seq_len-1]

    # Reshape logits and labels for CrossEntropyLoss
    # Flatten logits: [batch_size, seq_len-1, vocab_size] -> [batch_size * (seq_len-1), vocab_size]
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))

    # Flatten labels: [batch_size, seq_len-1] -> [batch_size * (seq_len-1)]
    flat_labels = shift_labels.view(-1)

    # Flatten loss mask: [batch_size, seq_len-1] -> [batch_size * (seq_len-1)]
    flat_mask = loss_mask.view(-1)

    # Compute Cross-Entropy Loss without reduction
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(flat_logits, flat_labels)  # Per-token loss: [batch_size * (seq_len-1)]

    # Apply the loss mask to exclude tokens not contributing to the loss
    loss = loss * flat_mask  # Element-wise multiplication with the mask

    # Normalize the total loss by the number of valid tokens
    valid_tokens = flat_mask.sum()  # Total number of tokens contributing to the loss
    if valid_tokens == 0:  # Prevent division by zero
        return torch.tensor(0.0, device=logits.device)
    return loss.sum() / valid_tokens

