import json
import os

import torch
import torch.nn.functional as F


def _get_tokenizer(clip_model):
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(clip_model.name_or_path)


def _encode_texts(clip_model, texts, device):
    tokenizer = _get_tokenizer(clip_model)
    tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    clip_model = clip_model.to(device)
    clip_model.eval()
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**tokens)
    return F.normalize(text_feat, dim=-1).detach()


def build_text_prototypes(clip_model, class_names, device, template="a photo of a {}, a type of aircraft"):
    """
    Encode the known-class names into frozen, L2-normalized text prototypes.
    class_names must already be ordered so row i corresponds to whatever
    local label index i the caller will use (e.g. position in sorted
    args.train_classes) -- this function does not know about class ids.

    Returns: [num_known, 512] tensor, requires_grad=False.
    """
    prompts = [template.format(name) for name in class_names]
    text_emb = _encode_texts(clip_model, prompts, device)
    text_emb.requires_grad_(False)
    return text_emb


def image_text_align_loss(img_proj, labels, text_emb, logit_scale=100.0):
    """
    CLIP-style InfoNCE between labeled image embeddings and their matching
    text prototype. `labels` must already be local indices into text_emb's
    rows (0..text_emb.size(0)-1), not raw dataset class ids.
    """
    img_proj = F.normalize(img_proj, dim=-1)
    logits = logit_scale * img_proj @ text_emb.t()
    return F.cross_entropy(logits, labels)


def select_cluster_exemplars(features, cluster_ids, k=8):
    """
    Offline helper: for each cluster, L2-normalize its member features and
    pick the top-k members nearest the cluster centroid (cosine similarity).

    features: [N, D] tensor or array-like of per-image features
    cluster_ids: [N] tensor or array-like of integer cluster assignments

    Returns: {cluster_id (int): [indices (int), ...]}
    """
    features = F.normalize(torch.as_tensor(features).float(), dim=-1)
    cluster_ids = torch.as_tensor(cluster_ids).long()

    exemplars = {}
    for cid in torch.unique(cluster_ids):
        idxs = torch.nonzero(cluster_ids == cid, as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        cluster_feats = features[idxs]
        centroid = F.normalize(cluster_feats.mean(dim=0, keepdim=True), dim=-1)
        sims = (cluster_feats @ centroid.t()).squeeze(-1)
        topk = min(k, idxs.numel())
        top_local = torch.topk(sims, topk).indices
        exemplars[int(cid.item())] = idxs[top_local].tolist()
    return exemplars


def load_pseudo_text_prototypes(clip_model, cache_path, num_clusters, device,
                                 template="a photo of a {}, a type of aircraft"):
    """
    Read a {"<cluster_id>": "<name>", ...} JSON cache written offline by
    make_pseudo_names.py and encode named clusters into text prototypes.
    Clusters missing from the cache, or with a blank/null name, get a zero
    row and valid=False. If cache_path doesn't exist, returns all-invalid.

    Returns: (proto [num_clusters, 512] normalized, valid [num_clusters] bool)
    """
    proj_dim = clip_model.config.projection_dim
    proto = torch.zeros(num_clusters, proj_dim, device=device)
    valid = torch.zeros(num_clusters, dtype=torch.bool, device=device)

    if not cache_path or not os.path.exists(cache_path):
        return proto, valid

    with open(cache_path, 'r') as f:
        names = json.load(f)

    named_ids, named_prompts = [], []
    for cid_str, name in names.items():
        cid = int(cid_str)
        if cid < 0 or cid >= num_clusters:
            continue
        if name is None or not str(name).strip():
            continue
        named_ids.append(cid)
        named_prompts.append(template.format(str(name).strip()))

    if len(named_ids) == 0:
        return proto, valid

    text_feat = _encode_texts(clip_model, named_prompts, device)
    for row, cid in enumerate(named_ids):
        proto[cid] = text_feat[row]
        valid[cid] = True

    return proto, valid


def pseudo_text_align_loss(img_proj, cluster_ids, cluster_conf, pseudo_proto, valid_mask,
                            conf_thresh=0.7, logit_scale=100.0):
    """
    Align unlabeled images to their cluster's pseudo-text prototype. Only
    samples whose predicted cluster has a name (valid_mask) AND whose
    cluster confidence clears conf_thresh contribute. The denominator is
    restricted to named clusters only, so unnamed (zero-row) clusters never
    pollute the softmax. Returns a scalar 0 tensor if nothing survives.
    """
    device = img_proj.device
    keep = valid_mask[cluster_ids] & (cluster_conf >= conf_thresh)
    if keep.sum() == 0:
        return torch.zeros((), device=device, dtype=img_proj.dtype)

    valid_ids = torch.nonzero(valid_mask, as_tuple=True)[0]
    id_to_col = torch.full((pseudo_proto.size(0),), -1, dtype=torch.long, device=device)
    id_to_col[valid_ids] = torch.arange(valid_ids.numel(), device=device)

    kept_img = F.normalize(img_proj[keep], dim=-1)
    kept_labels = id_to_col[cluster_ids[keep]]
    valid_proto = pseudo_proto[valid_ids]

    logits = logit_scale * kept_img @ valid_proto.t()
    return F.cross_entropy(logits, kept_labels)
