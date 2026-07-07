"""
Offline cluster-naming scaffold. Run this standalone on a login node WITH
internet access -- NOT inside a Compute Canada compute job, which has none.

    python make_pseudo_names.py --model_path dev_outputs/.../checkpoints/model.pt \
        --dataset_name aircraft --output pseudo_names.json

It loads a trained (or mid-training) vit_clip SEAL checkpoint, extracts
features + cluster assignments for the UNLABELED split, picks exemplar
images per cluster, and writes a cluster_id -> name JSON stub that
train_seal.py's --pseudo_names_path / Stage 2 pseudo-text alignment reads.

The actual naming (showing exemplars to a VLM/LLM and asking for a coarse
manufacturer/family-level name) is deliberately left as a TODO below --
this script only wires up the plumbing around it.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.augmentations import get_transform
from data.get_datasets import get_datasets_v2, get_class_splits
from model import vit_threeHeads_v2, vit_twoHeads_v2
from clip_backbone import build_clip_backbone
from text_align import select_cluster_exemplars

from birds_category import trees as birds_category_list
from aircraft_category import trees as aircraft_category_list
from cars_category import trees as cars_category_list

two_level_datasets = ['scars']


def build_model(args):
    backbone, clip_model = build_clip_backbone(args.clip_model_name)

    if args.dataset_name == 'cub':
        num_superclass = max([i[1] for i in birds_category_list])
        num_fine = max([i[2] for i in birds_category_list])
    elif args.dataset_name == 'aircraft':
        num_superclass = max([i[2] for i in aircraft_category_list])
        num_fine = max([i[1] for i in aircraft_category_list])
    elif args.dataset_name == 'scars':
        num_superclass = max([i[1] for i in cars_category_list])
        num_fine = 0
    else:
        raise ValueError("Not Support for this dataset")

    mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    if args.dataset_name in two_level_datasets:
        model = vit_twoHeads_v2(backbone=backbone, in_dim=768, num_class=num_fine,
                                 num_superclass=num_superclass, num_fine=mlp_out_dim,
                                 nlayers=3, feature_size=768)
    else:
        model = vit_threeHeads_v2(backbone=backbone, in_dim=768, num_class=num_fine,
                                   num_superclass=num_superclass, num_fine=mlp_out_dim,
                                   nlayers=3, feature_size=768)

    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


@torch.no_grad()
def extract_unlabelled_clusters(model, unlabelled_loader):
    features, cluster_ids, uq_idxs = [], [], []
    for images, _, batch_uq_idxs in unlabelled_loader:
        images = images.cuda(non_blocking=True)
        (_, _), (_, _), (species_proj, species_out) = model(images)
        features.append(species_proj.cpu())
        cluster_ids.append(species_out.argmax(dim=1).cpu())
        uq_idxs.append(batch_uq_idxs)
    return torch.cat(features), torch.cat(cluster_ids), torch.cat(uq_idxs)


def main():
    parser = argparse.ArgumentParser(description='offline cluster naming (login node, needs internet)')
    parser.add_argument('--dataset_name', type=str, default='aircraft')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--interpolation', type=int, default=3)
    parser.add_argument('--crop_pct', type=float, default=0.875)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--model_path', type=str, required=True, help='path to a saved SEAL checkpoint (vit_clip)')
    parser.add_argument('--output', type=str, default='pseudo_names.json')
    parser.add_argument('--exemplars_k', type=int, default=8)
    args = parser.parse_args()

    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    num_clusters = args.num_labeled_classes + args.num_unlabeled_classes

    _, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    _, _, unlabelled_train_examples_test, _, _, _ = get_datasets_v2(
        args.dataset_name, test_transform, test_transform, args)

    unlabelled_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                    batch_size=args.batch_size, shuffle=False, pin_memory=False)

    model = build_model(args)
    features, cluster_ids, uq_idxs = extract_unlabelled_clusters(model, unlabelled_loader)

    exemplars = select_cluster_exemplars(features, cluster_ids, k=args.exemplars_k)

    pseudo_names = {}
    for cluster_id in range(num_clusters):
        if cluster_id not in exemplars:
            # No unlabeled samples were assigned to this cluster (e.g. still
            # empty early in training) -- leave it unnamed.
            pseudo_names[str(cluster_id)] = ""
            continue

        exemplar_local_idxs = exemplars[cluster_id]
        exemplar_uq_idxs = uq_idxs[exemplar_local_idxs].tolist()
        exemplar_paths = [unlabelled_train_examples_test.samples[i][0] for i in exemplar_local_idxs]

        # ------------------------------------------------------------
        # TODO(you): call your VLM/LLM naming backend here.
        #   - Show it exemplar_paths (k images nearest this cluster's centroid)
        #   - Ask for a COARSE manufacturer/family-level name (not a fine
        #     variant guess -- e.g. "Boeing 737" rather than "737-800"),
        #     since this is cluster-then-name on unlabeled/novel clusters,
        #     not zero-shot CLIP classification.
        #   - Leave the name "" (blank) if the VLM isn't confident; blank/
        #     missing names are treated as unnamed (valid=False) by
        #     text_align.load_pseudo_text_prototypes.
        #
        # name = call_vlm_naming_backend(exemplar_paths)
        # ------------------------------------------------------------
        name = ""

        pseudo_names[str(cluster_id)] = name
        print(f'cluster {cluster_id}: {len(exemplar_paths)} exemplars (uq_idxs={exemplar_uq_idxs}) -> name={name!r}')

    with open(args.output, 'w') as f:
        json.dump(pseudo_names, f, indent=2)
    print(f'Wrote {len(pseudo_names)} cluster entries to {args.output}')


if __name__ == '__main__':
    main()
