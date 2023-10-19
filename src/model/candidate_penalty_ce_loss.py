import torch.nn.functional as F
import torch

def get_normalized_probs(net_output, log_probs):
    return get_normalized_probs_scriptable(net_output, log_probs)

def get_normalized_probs_scriptable(net_output, log_probs):
    logits = net_output.float()
    return F.log_softmax(logits, dim=-1)

def candidate_penalty_cross_entropy_criterion(net_output, target, tokenizer , reduce=True, compute_custom_metrics=True):
    nsentences = target.size(1)
    target = target.view(-1)
    rank_alpha = 1.0

    # -- mle loss
    lprobs = get_normalized_probs(net_output, log_probs=True)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    true_token_lprobs = F.nll_loss(
        lprobs,
        target,
        ignore_index=tokenizer.pad_token_id,
        reduction='none',
    )
    mle_loss = true_token_lprobs.sum()

    with torch.no_grad():
        ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
        ctx_cands_ = (ctx_cands.tril(-1) + tokenizer.pad_token_id)
        ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
        ctx_cands = ctx_cands.tril(-1) + ctx_cands_
        ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), tokenizer.pad_token_id)
        negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)

    one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)

    custom_loss = -torch.log(one_minus_probs) * negative_targets
    custom_loss = custom_loss.sum()

    loss = mle_loss + rank_alpha * custom_loss
    return loss

