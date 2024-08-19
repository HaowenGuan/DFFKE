import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_loss(output, target, loss_function):
    """
    Calculate the loss between the output and target.

    @param output: Must be probabilistic form (e.g. softmax)
    @param target: Must be probabilistic form (e.g. softmax), and must be detached from gradient diagram
    @param loss_function: Loss function to use
    @return: Loss value
    """
    MSELoss = nn.MSELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    if loss_function == 'MSE':
        loss = MSELoss(output, target)
    elif loss_function == 'KLDiv':
        # Note the input to KLDivLoss need to be log probability
        loss = KLDivLoss(torch.log(output), target)
    else:
        raise ValueError(f'Loss function {loss_function} not implemented')

    return loss


def contrastive(
        a: torch.Tensor,
        b: torch.Tensor,
        a_label: torch.Tensor = None,
        b_label: torch.Tensor = None,
        temperature_matrix=None,
):
    """
    Calculate the contrastive loss between two sets of embeddings.

    @param a: torch.Tensor, shape (..., m, d)
    @param b: torch.Tensor, shape (..., n, d)
    @param a_label: torch.Tensor,（Optional) shape (m,)
    @param b_label: torch.Tensor,（Optional) shape (n,)
    @param temperature_matrix: torch.Tensor, shape (n,)
    @return: torch.Tensor, shape (1,)
    """
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        return torch.matmul(h1, h2.transpose(-1, -2))


    assert a.shape[:-2] == b.shape[:-2]
    assert a.shape[-1] == b.shape[-1]
    if a_label is not None and b_label is not None:
        # (m, n)
        pos_mask = (a_label.view(-1, 1) == b_label.view(1, -1)).float()
    else:
        pos_mask = torch.eye(a.shape[-2], dtype=torch.float).to(a.device)
    # (1, ..., 1, m, n)
    pos_mask = pos_mask.view((1,) * (a.dim() - 2) + pos_mask.shape)
    # (..., m, n)
    sim = _similarity(a, b / temperature_matrix if temperature_matrix else b)
    exp_sim = torch.exp(sim)
    # (..., m)
    target = (exp_sim * pos_mask).sum(dim=-1)
    # (..., m)
    prob = target / (exp_sim.sum(dim=-1) + 1e-9)
    log_prob = torch.log(prob)
    loss = -log_prob.mean()
    return loss


def inverse_cross_entropy(
    logit: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'):
    """
    Inverse cross entropy loss for model discrepancy.
    
    @param logit: model output before softmax
    @param target: target in probabilistic form
    @param reduction: 'mean', 'sum', or 'none'
    """
    prob = F.softmax(logit, dim=-1)
    inv_prob = 1 - prob
    log_inv_prob = torch.log(inv_prob + 1e-9)
    weighted_sum_log_inv_prob =  torch.sum(target * log_inv_prob, dim=-1)
    if reduction == 'mean':
        return -torch.mean(weighted_sum_log_inv_prob)
    if reduction == 'sum':
        return -torch.sum(weighted_sum_log_inv_prob)
    return -weighted_sum_log_inv_prob


