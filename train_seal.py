import argparse

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, get_datasets_v2

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root,dino_pretrain_path, dinov2_pretrain_path, aircraft_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, vit_threeHeads_v2, vit_twoHeads_v2, info_nce_logits_smooth
import vision_transformer as vits
import vision_transformers_v2 as vits_v2
from clip_backbone import build_clip_backbone
from text_align import build_text_prototypes, image_text_align_loss, load_pseudo_text_prototypes, pseudo_text_align_loss
import gc
from birds_category import trees as birds_category_list
from birds_category import get_order_family_target as get_birds_order_family_target
from aircraft_category import trees as aircraft_category_list
from aircraft_category import get_order_family_target as get_aircraft_order_family_target

from cars_category import trees as cars_category_list
from cars_category import get_order_family_target as get_cars_order_family_target





two_level_datasets = ['scars']
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity,smoothing = 0.5):
        target_smooth = F.one_hot(target,input.size(1)).float()*(1-smoothing) +smoothing*similarity
        return torch.nn.CrossEntropyLoss()(input, target_smooth)



def hierarchical_similarity(f_order, f_family, f_species, alpha=0.6, beta=0.3, gamma=0.1, method = 'cos'):
    """
    f_order, f_family, f_species: [batch_size, feature_dim]
    return: similarity [batch_size, batch_size]
    """

    # Normalize features
    f_order = F.normalize(f_order.detach(), dim=-1)
    if f_family is not None:
        f_family = F.normalize(f_family.detach(), dim=-1)
    f_species = F.normalize(f_species.detach(), dim=-1)

    # Compute cosine similarities
    if method == 'cos':
        sim_order = torch.matmul(f_order, f_order.T)          # [batch_size, batch_size]
        if f_family is not None:
            sim_family = torch.matmul(f_family, f_family.T)
        sim_species = torch.matmul(f_species, f_species.T)
    else:
        sim_order = -torch.cdist(f_order, f_order)          # [batch_size, batch_size]
        if f_family is not None:
            sim_family = -torch.cdist(f_family, f_family)
        sim_species =-torch.cdist(f_species, f_species)

    # Weighted combination
    if f_family is not None:
        sim_final_3 = alpha * sim_species + beta * sim_family + gamma * sim_order
        sim_final_2 = beta/(beta + gamma)  * sim_family + gamma/(beta + gamma) * sim_order
        sim_final_1 = sim_order
        
        sim_final_2 = (sim_final_2 - sim_final_2.min()) / (sim_final_2.max() - sim_final_2.min() + 1e-10)
        sim_final_2 = sim_final_2 / sim_final_2.sum(dim=1)
    else:
        sim_final_3 = alpha * sim_species  + gamma * sim_order
        sim_final_1 = sim_order
        sim_final_2 = None
    
    # Normalize similarity to [0,1] if desired
    sim_final_3 = (sim_final_3 - sim_final_3.min()) / (sim_final_3.max() - sim_final_3.min() + 1e-10)
    sim_final_3 = sim_final_3 / sim_final_3.sum(dim=1)
    # Normalize similarity to [0,1] if desired
    sim_final_1 = (sim_final_1 - sim_final_1.min()) / (sim_final_1.max() - sim_final_1.min() + 1e-10)
    sim_final_1 = sim_final_1 / sim_final_1.sum(dim=1)
    

    return sim_final_1, sim_final_2, sim_final_3



def get_known_class_names(args):
    """
    Returns known-class display names ordered to match sorted(args.train_classes),
    i.e. row i is the name of the class at local index i (see args.known_id_lut
    in __main__, which maps global dataset class ids to this same local index).

    Currently only implemented for the plain 'aircraft' (variant-level) task:
    reads FGVC-Aircraft's own images_variant_trainval.txt and reproduces the
    exact class-id ordering FGVCAircraft.find_classes() uses (np.unique over
    the raw label strings), so indices line up with args.train_classes without
    needing to instantiate the full dataset.
    """
    if args.dataset_name == 'aircraft':
        classes_file = os.path.join(aircraft_root, 'data', 'images_variant_trainval.txt')
        targets = []
        with open(classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                targets.append(' '.join(split_line[1:]).strip())
        all_names = np.unique(targets)  # class id i -> all_names[i]
    else:
        raise NotImplementedError(f'get_known_class_names not implemented for dataset_name={args.dataset_name}')

    known_sorted = sorted(args.train_classes)
    return [all_names[i] for i in known_sorted]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(student, train_loader, test_loader, unlabelled_train_loader, args, get_order_family_target, test_loader_all):
 
    
    params_list = []
    params_groups_backbone = get_params_groups(student.module.backbone)
    params_groups_backbone[0]['lr'] = args.backbone_lr
    params_groups_backbone[1]['lr'] = args.backbone_lr
    params_list.extend(params_groups_backbone)
    params_groups_features = get_params_groups(student.module.features)
    params_groups_features[0]['lr'] = args.features_lr
    params_groups_features[1]['lr'] = args.features_lr
    params_list.extend(params_groups_features)
    params_groups_projector_1 = get_params_groups(student.module.projector_super)
    params_groups_projector_1[0]['lr'] = args.projector_1_lr
    params_groups_projector_1[1]['lr'] = args.projector_1_lr
    params_list.extend(params_groups_projector_1)
    params_groups_projector_3 = get_params_groups(student.module.projector_fine)
    params_groups_projector_3[0]['lr'] = args.projector_3_lr
    params_groups_projector_3[1]['lr'] = args.projector_3_lr
    params_list.extend(params_groups_projector_3)
    if args.dataset_name not in two_level_datasets:
        params_groups_projector_2 = get_params_groups(student.module.projector_class)
        params_groups_projector_2[0]['lr'] = args.projector_2_lr
        params_groups_projector_2[1]['lr'] = args.projector_2_lr
        params_list.extend(params_groups_projector_2)
    
    if args.dataset_name not in two_level_datasets:
        optimizer = SGD(params_list, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(params_list, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    cluster_criterions = [DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    ) for i in range(3)]

    known_families = set()
    known_orders = set()
    if args.dataset_name not in two_level_datasets:
        M_species_family = torch.zeros(args.num_species, args.num_families)
        M_family_order = torch.zeros(args.num_families, args.num_orders)
        for species_idx in args.train_classes:

            order_idx, family_idx = get_order_family_target([species_idx])
            order_idx, family_idx = order_idx.cpu().numpy()[0], family_idx.cpu().numpy()[0]
            M_species_family[species_idx, family_idx] = 1.0
            M_family_order[family_idx, order_idx] = 1.0
            known_families.add(family_idx)
            known_orders.add(order_idx)
        for species_idx in [i for i in range(args.num_species) if i not in args.train_classes]:
            M_species_family[species_idx] = torch.ones(args.num_families) / args.num_families
        for family_idx in range(args.num_families):
            if family_idx not in known_families:
                M_family_order[family_idx] = torch.ones(args.num_orders) / args.num_orders
        M_species_family = M_species_family.cuda()
        M_family_order = M_family_order.cuda()
    else:
        M_species_family = torch.zeros(args.num_species, args.num_orders)
        for species_idx in args.train_classes:
 
            order_idx = get_order_family_target([species_idx])
            order_idx = order_idx.cpu().numpy()[0]
            M_species_family[species_idx, order_idx] = 1.0
            known_orders.add(order_idx)
        for species_idx in [i for i in range(args.num_species) if i not in args.train_classes]:
            M_species_family[species_idx] = torch.ones(args.num_orders) / args.num_orders
        M_species_family = M_species_family.cuda()
        M_family_order = None
    args.known_families = known_families
    args.known_orders = known_orders

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        
        sup_con_loss_records = [AverageMeter() for i in range(3)]
        cluster_loss_records = [AverageMeter() for i in range(3)]
        contrastive_loss_records = [AverageMeter() for i in range(3)]
        cls_loss_records = [AverageMeter() for i in range(3)]
        consistency_loss_1_recorder = AverageMeter()
        consistency_loss_2_recorder = AverageMeter()
        
        train_acc_record = AverageMeter()

        if (getattr(args, 'vit_clip', False) and args.pseudo_align_weight > 0
                and args.pseudo_refresh_epochs > 0 and epoch > 0
                and epoch % args.pseudo_refresh_epochs == 0
                and os.path.exists(args.pseudo_names_path)):
            args.pseudo_proto, args.pseudo_valid = load_pseudo_text_prototypes(
                args.clip_model, args.pseudo_names_path, args.num_species, args.device)
            args.logger.info(f'[epoch {epoch}] reloaded pseudo text prototypes from {args.pseudo_names_path}')

        student.train()
        
       
        memax_list =[args.memax_weight_1, args.memax_weight_2, args.memax_weight]
       
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            
            if args.dataset_name not in two_level_datasets:
                order_targets, family_targets= get_order_family_target(class_labels)
                order_targets, family_targets = order_targets.cuda(non_blocking=True), family_targets.cuda(non_blocking=True)
            else:
                order_targets, family_targets= get_order_family_target(class_labels), None
                order_targets = order_targets.cuda(non_blocking=True)
            
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            labels = [order_targets, family_targets, class_labels]
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                loss = 0
                
                # (order_proj, order_out), (family_proj, family_out), (species_proj, species_out) = student(images)
                outputs =  student(images)
                
                sim_final_1, sim_final_2, sim_final_3 = hierarchical_similarity(outputs[0][0], outputs[1][0], outputs[2][0], args.sim_alpha, args.sim_beta, args.sim_gamma)
                sim_final = [sim_final_1, sim_final_2, sim_final_3]
                sim_final_1_dist, sim_final_2_dist, sim_final_3_dist = hierarchical_similarity(outputs[0][0], outputs[1][0], outputs[2][0], args.sim_alpha, args.sim_beta, args.sim_gamma, method = 'euc')
                sim_final_dist = [sim_final_1_dist, sim_final_2_dist, sim_final_3_dist]
                total_cluster_loss = 0
                total_con_loss = 0
                for level in range(3):
                    if args.dataset_name in two_level_datasets and level == 1:
                        continue
                    cluster_criterion = cluster_criterions[level]
                    targets = labels[level]
                    student_proj, student_out = outputs[level]
                    teacher_out = student_out.detach()

                    # clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([targets[mask_lab] for _ in range(2)], dim=0)
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                    # clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    cluster_loss += memax_list[level] * me_max_loss

                   
                    
                    # angle-based
                    contrastive_logits, contrastive_labels, sim = info_nce_logits_smooth(features=student_proj, confusion_factor=sim_final[level], args=args)
     
            
                    contrastive_loss_angle = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, sim, args.unsupervised_smoothing)
                
                    # distance-based
                    contrastive_logits_dis, contrastive_labels_dis, sim_dist = info_nce_logits_smooth(features=student_proj, confusion_factor=sim_final_dist[level], args=args, similarity='euc')
                    contrastive_loss_dis = LabelSmoothingLoss()(contrastive_logits_dis, contrastive_labels_dis, sim_dist,  args.unsupervised_smoothing)
                    
                    lambda_dis = (epoch - (args.hyper_start_epoch - 1)) / ((args.hyper_end_epoch - 1) - (args.hyper_start_epoch - 1))
                    lambda_dis = torch.max(torch.tensor([0, lambda_dis])).item()
                    lambda_dis = torch.min(torch.tensor([1, lambda_dis])).item()
                    contrastive_loss = (1 - lambda_dis) * contrastive_loss_angle + lambda_dis * contrastive_loss_dis
                    
                   
                    # representation learning, sup
                    student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                    student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                    sup_con_labels = targets[mask_lab]
           
                    sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                    total_cluster_loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    total_con_loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                    
                    sup_con_loss_records[level].update(sup_con_loss.item(), targets.size(0))
                    cluster_loss_records[level].update(cluster_loss.item(), targets.size(0))
                    contrastive_loss_records[level].update(contrastive_loss.item(), targets.size(0))
                    cls_loss_records[level].update(cls_loss.item(), targets.size(0))


    
                loss += total_cluster_loss
                loss += total_con_loss
                
      
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                # Train acc
                if args.dataset_name not in two_level_datasets:
                    (order_proj, order_out), (family_proj, family_out), (species_proj, species_out) = outputs
                    
                    # species_out_label = species_out.argmax(1)
                    # mask_novel = torch.tensor([True if x.item() in args.train_classes else False for x in species_out_label]).cuda()
                  

                    p_order = F.softmax(order_out / args.kl_temp, dim=-1)
                    p_family = F.softmax(family_out  / args.kl_temp, dim=-1)
                    p_species = F.softmax(species_out  / args.kl_temp, dim=-1)
                    
                    inferred_family_from_species = p_species @ M_species_family 
                    inferred_order_from_family = p_family @ M_family_order 
                    
                    kl_loss_species_family = F.kl_div(p_family.log(), inferred_family_from_species, reduction='batchmean')
                    kl_loss_family_order = F.kl_div(p_order.log(), inferred_order_from_family, reduction='batchmean')
                    
                
                    
                    inferred_family_from_order = p_order @ M_family_order.T        
                    inferred_species_from_family = p_family @ M_species_family.T   

                    kl_loss_order_family = F.kl_div(p_family.log(), inferred_family_from_order, reduction='batchmean')
                    kl_loss_family_species = F.kl_div(p_species.log(), inferred_species_from_family, reduction='batchmean')

                else:
                    (order_proj, order_out), (family_proj, family_out), (species_proj, species_out) = outputs
                    
                    
                    p_order = F.softmax(order_out / args.kl_temp, dim=-1)
                    p_species = F.softmax(species_out  / args.kl_temp, dim=-1)
                    
                    inferred_order_from_species = p_species @ M_species_family 
                    kl_loss_family_order = F.kl_div(p_order.log(), inferred_order_from_species, reduction='batchmean')
                    kl_loss_species_family = 0.0
                    inferred_species_from_order = p_order @ M_species_family.T 
                    kl_loss_order_family = F.kl_div(p_species.log(), inferred_species_from_order, reduction='batchmean')
                    kl_loss_family_species = 0.0
        
                
                loss += args.kl_weight*kl_loss_species_family
                loss += args.kl_weight*kl_loss_family_order
                
                
                pstr += f'kl_loss_family_order: {kl_loss_family_order.item():.4f} '
                if args.dataset_name not in two_level_datasets:
                    pstr += f'kl_loss_species_family: {kl_loss_species_family.item():.4f} '

                # ----------------------------------------------------------------
                # CLIP image-text alignment (Stage 1: labeled, Stage 2: pseudo-text)
                # Both gated behind their weight args, which default to 0.0, so the
                # baseline (vit_dino / vit_dino_v2, or vit_clip with weights at 0.0)
                # runs bit-for-bit identical to before -- no extra backbone forward
                # passes are issued, so no extra RNG (dropout) is consumed either.
                # ----------------------------------------------------------------
                vit_clip = getattr(args, 'vit_clip', False)
                if vit_clip and (args.align_weight > 0 or
                                  (args.pseudo_align_weight > 0 and args.pseudo_proto is not None)):
                    img_views = images.chunk(2, dim=0)

                if vit_clip and args.align_weight > 0:
                    labeled_images = torch.cat([v[mask_lab] for v in img_views], dim=0)
                    labeled_targets = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                    local_labels = args.known_id_lut[labeled_targets]
                    _, img_proj = student.module.backbone(labeled_images, return_proj=True)
                    align_loss = image_text_align_loss(img_proj, local_labels, args.text_emb)
                    loss += args.align_weight * align_loss
                    pstr += f'align_loss: {align_loss.item():.4f} '

                    if args.debug and epoch == 0 and batch_idx == 0:
                        with torch.no_grad():
                            sample_n = min(5, img_proj.size(0))
                            sims = F.normalize(img_proj[:sample_n], dim=-1) @ args.text_emb.t()
                            args.logger.info(f'[debug] image->text similarity (first {sample_n} labeled samples):')
                            for i in range(sample_n):
                                true_idx = local_labels[i].item()
                                pred_idx = sims[i].argmax().item()
                                true_name = args.known_class_names[true_idx]
                                pred_name = args.known_class_names[pred_idx]
                                args.logger.info(
                                    f'  sample {i}: true={true_name!r} top1_pred={pred_name!r} '
                                    f'top1_sim={sims[i, pred_idx].item():.4f} true_sim={sims[i, true_idx].item():.4f}')

                if vit_clip and args.pseudo_align_weight > 0 and args.pseudo_proto is not None:
                    unlab_images = torch.cat([v[~mask_lab] for v in img_views], dim=0)
                    species_chunks = species_out.chunk(2, dim=0)
                    unlab_cluster_logits = torch.cat([c[~mask_lab] for c in species_chunks], dim=0)
                    unlab_cluster_probs = F.softmax(unlab_cluster_logits, dim=-1)
                    unlab_cluster_conf, unlab_cluster_ids = unlab_cluster_probs.max(dim=-1)
                    _, unlab_img_proj = student.module.backbone(unlab_images, return_proj=True)
                    pseudo_loss = pseudo_text_align_loss(
                        unlab_img_proj, unlab_cluster_ids, unlab_cluster_conf,
                        args.pseudo_proto, args.pseudo_valid, conf_thresh=args.conf_thresh)
                    loss += args.pseudo_align_weight * pseudo_loss
                    pstr += f'pseudo_align_loss: {pseudo_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
        
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        
        del loss, order_proj,family_proj,species_proj 
        gc.collect()
        torch.cuda.empty_cache()
        
        
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc,  M_species_family,M_family_order   = test_updateM(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args, get_order_family_target = get_order_family_target, M_species_family=M_species_family,M_family_order=M_family_order)
        args.logger.info('Testing on disjoint test set...')
        # test(model, test_loader, epoch, save_name, args, get_order_family_target):
        all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args, get_order_family_target=get_order_family_target)
    
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))


        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))
        
        
       
      


def test_updateM(model, test_loader, epoch, save_name, args, get_order_family_target, M_species_family, M_family_order):
    update_thd = args.update_thd
    model.eval()

    preds, targets = [], []
    preds_order, preds_family = [], []
    orders, familys = [], []
    mask = np.array([])
    logits_species = []
    logits_family = []
    logits_order = []
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        if args.dataset_name in two_level_datasets:
            order = get_order_family_target(label)
        else:
            order, family = get_order_family_target(label)
        with torch.no_grad():
            (order_proj, order_out), (family_proj, family_out), (species_proj, species_out) = model(images)
            
            if args.dataset_name not in two_level_datasets:
                    
                logits_species.append(species_out)
                logits_family.append(family_out)
                logits_order.append(order_out)
            else:
                logits_species.append(species_out)
                logits_order.append(order_out)
            
           
            
            preds.append(species_out.argmax(1).cpu().numpy())

            
            targets.append(label.cpu().numpy())
    
         
            
            mask = np.append(mask, np.array([True if x.item() in args.train_classes else False for x in label]))
    logits_species = torch.concatenate(logits_species)
    logits_order = torch.concatenate(logits_order)
    if args.dataset_name not in two_level_datasets:
        logits_family = torch.concatenate(logits_family)
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    

    
    if epoch < args.warmup_epoch_matrix:
        M_momentum = 0
    else:
        M_momentum = args.M_momentum
    if args.dataset_name not in two_level_datasets:
        species_probs = F.softmax(logits_species, dim=-1)  # [128, num_species]
        family_probs = F.softmax(logits_family, dim=-1)    # [128, num_families]
        for species_idx in [i for i in range(args.num_species) if i not in args.train_classes]:
            species_conf, species_pred = species_probs.max(dim=1)
                
            species_mask = (species_pred == species_idx)  & (species_conf > update_thd) # 找到属于该species的样本
            if species_mask.sum() > 0:
                avg_family_prob = family_probs[species_mask].mean(dim=0)
                # momentum = 0.9
                M_species_family[species_idx] = (
                    M_momentum * M_species_family[species_idx] + (1 - M_momentum ) * avg_family_prob
                )
                M_species_family[species_idx] /= M_species_family[species_idx].sum()
        for family_idx in range(args.num_families):
            if family_idx not in args.known_families:
                family_conf, family_pred = family_probs.max(dim=1)
                family_mask = (family_pred == family_idx) & (family_conf >  update_thd)
                if family_mask.sum() > 0:
                    avg_order_prob = F.softmax(logits_order[family_mask], dim=-1).mean(dim=0)
                
                    M_family_order[family_idx] = (
                        M_momentum * M_family_order[family_idx] + (1 - M_momentum) * avg_order_prob
                    )
                    M_family_order[family_idx] /= M_family_order[family_idx].sum()
    else:
        order_probs = F.softmax(logits_order, dim=-1) 
        species_probs = F.softmax(logits_species, dim=-1) 
        species_conf, species_pred = species_probs.max(dim=1)
        for species_idx in [i for i in range(args.num_species) if i not in args.train_classes]:
            
            species_mask = (species_pred == species_idx) 
            if species_mask.sum() > 0:
                avg_family_prob = order_probs[species_mask].mean(dim=0)
                M_species_family[species_idx] = avg_family_prob
        
                    
                M_species_family[species_idx] = (
                        M_momentum * M_species_family[species_idx] + (1 - M_momentum) * avg_family_prob
                    )
                M_species_family[species_idx] /= M_species_family[species_idx].sum()
    
    return all_acc, old_acc, new_acc,  M_species_family, M_family_order






def test(model, test_loader, epoch, save_name, args, get_order_family_target):

    model.eval()

    preds, targets = [], []

    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        if args.dataset_name in two_level_datasets:
            order = get_order_family_target(label)
        else:
            order, family = get_order_family_target(label)
        with torch.no_grad():
            (order_proj, order_out), (family_proj, family_out), (species_proj, species_out) = model(images)
            
            preds.append(species_out.argmax(1).cpu().numpy())
        
            
            targets.append(label.cpu().numpy())
            
       
            
            mask = np.append(mask, np.array([True if x.item() in args.train_classes else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1','v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--memax_weight_1', type=float, default=0.5)
    parser.add_argument('--memax_weight_2', type=float, default=0.5)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    
    parser.add_argument('--random_seed', default=666, type=int)
    parser.add_argument('--model_name', default='vit_dino', type=str, help='options: vit_dino, vit_dino_v2, vit_clip')

    # ----------------------
    # CLIP TEXT ALIGNMENT (all default to off; baseline is unchanged at defaults)
    # ----------------------
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch16',
                         help='HuggingFace CLIP checkpoint used when --model_name vit_clip')
    parser.add_argument('--align_weight', type=float, default=0.0,
                         help='Stage 1: weight for labeled image<->known-class-text alignment loss')
    parser.add_argument('--pseudo_names_path', type=str, default='',
                         help='Stage 2: path to offline cluster_id->name JSON cache (see make_pseudo_names.py)')
    parser.add_argument('--pseudo_align_weight', type=float, default=0.0,
                         help='Stage 2: weight for unlabeled image<->pseudo-cluster-text alignment loss')
    parser.add_argument('--conf_thresh', type=float, default=0.7,
                         help='Stage 2: minimum cluster softmax confidence required to use a pseudo-text pair')
    parser.add_argument('--pseudo_refresh_epochs', type=int, default=0,
                         help='Stage 2: reload --pseudo_names_path every N epochs (0 = load once, never refresh)')
    parser.add_argument('--debug', action='store_true', default=False,
                         help='print an image->text similarity sanity check for a few labeled samples')

    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)

    parser.add_argument('--feature_size', default=768, type=int)
    parser.add_argument('--update_thd', type=float, default=0.0)
    parser.add_argument('--M_momentum', type=float, default=0.9)
    parser.add_argument('--kl_temp', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    
    parser.add_argument('--sim_alpha', type=float, default=0.3)
    parser.add_argument('--sim_beta', type=float, default=0.3)
    parser.add_argument('--sim_gamma', type=float, default=0.3)
    
    parser.add_argument('--backbone_lr', type=float, default=0.1)
    parser.add_argument('--features_lr', type=float, default=0.1)
    parser.add_argument('--projector_1_lr', type=float, default=0.1)
    parser.add_argument('--projector_2_lr', type=float, default=0.1)
    parser.add_argument('--projector_3_lr', type=float, default=0.1)
    
    parser.add_argument('--unsupervised_smoothing', type=float, default=0.5)
    
    parser.add_argument('--P_momentum', type=float, default=0.9)
    parser.add_argument('--warmup_epoch_matrix', default=30, type=int, help='warmup epoch for matrix momentum update')

    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    print(args)
    set_random_seed(args.random_seed)
    device = torch.device('cuda:0')
    args.device=device
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'simgcd_baseline'])
    args.logger.info(f'Using evaluation function {args.eval_funcs} to print results')
    
    # torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768 
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    args.vit_clip = (args.model_name == 'vit_clip')

    if args.model_name == 'vit_dino':
        backbone = vits.__dict__['vit_base']()

        state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    elif args.model_name == 'vit_dino_v2':
        backbone = vits_v2.__dict__['vit_base']()
        state_dict = torch.load(dinov2_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    elif args.model_name == 'vit_clip':
        # backbone: CLIPVisionWrapper (768-d pooled features by default, matching
        # the DINO backbones' contract). args.clip_model is the full CLIPModel
        # (vision + text) kept around so the frozen text encoder is usable below
        # and in the training loop for image-text alignment.
        backbone, args.clip_model = build_clip_backbone(args.clip_model_name)
    else:
        raise ValueError('Invalid model name')
    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))



    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    if args.vit_clip:
        # CLIPVisionWrapper uses HF's naming ('vision_model.encoder.layers.N...')
        # rather than DINO's ('blocks.N...'), so it needs its own block match.
        for name, m in backbone.named_parameters():
            if 'encoder.layers.' in name:
                block_num = int(name.split('encoder.layers.')[1].split('.')[0])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    else:
        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True


    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, train_dataset_test, labelled_train_examples_test = get_datasets_v2(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # model = nn.Sequential(backbone, projector).to(device)
    if args.dataset_name == 'cub':
        num_superclass = max([i[1] for i in birds_category_list])
        num_fine = max([i[2] for i in birds_category_list])
    elif args.dataset_name == 'aircraft':
        num_superclass = max([i[2] for i in aircraft_category_list])
        num_fine = max([i[1] for i in aircraft_category_list])
    elif args.dataset_name  == 'scars':
        num_superclass = max([i[1] for i in cars_category_list])
        num_fine = 0
    else:
        raise ValueError("Not Support for this dataset")
    
    get_order_family_target_dict = {
        'cub': get_birds_order_family_target,
        'aircraft': get_aircraft_order_family_target,
        'scars': get_cars_order_family_target,
    }
    
    if args.dataset_name in two_level_datasets:
        model = vit_twoHeads_v2(backbone=backbone,in_dim= args.feat_dim, num_class=num_fine,num_superclass = num_superclass,num_fine=args.mlp_out_dim, nlayers=args.num_mlp_layers, feature_size = args.feature_size)
    else:
        model = vit_threeHeads_v2(backbone=backbone,in_dim= args.feat_dim, num_class=num_fine,num_superclass = num_superclass,num_fine=args.mlp_out_dim, nlayers=args.num_mlp_layers, feature_size = args.feature_size)
    model = nn.DataParallel(model) 
    model = model.cuda()
    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    args.num_species = args.num_labeled_classes + args.num_unlabeled_classes
    args.num_families = num_fine
    args.num_orders = num_superclass

    # ----------------------
    # CLIP TEXT ALIGNMENT SETUP (Stage 1 labeled, Stage 2 pseudo-text)
    # Everything here is a no-op unless --model_name vit_clip and the
    # corresponding weight arg is > 0.
    # ----------------------
    args.text_emb = None
    args.known_id_lut = None
    args.known_class_names = None
    args.pseudo_proto = None
    args.pseudo_valid = None

    if args.vit_clip:
        # Text encoder is frozen and never joins the optimizer: it is not part
        # of `backbone` (see clip_backbone.CLIPVisionWrapper), so student.train()
        # / get_params_groups(student.module.backbone) never touch it.
        for p in args.clip_model.text_model.parameters():
            p.requires_grad = False
        for p in args.clip_model.text_projection.parameters():
            p.requires_grad = False
        args.clip_model.text_model.eval()
        args.clip_model.text_projection.eval()

        if args.align_weight > 0:
            known_sorted = sorted(args.train_classes)
            lut = torch.full((args.num_species,), -1, dtype=torch.long)
            for local_idx, global_id in enumerate(known_sorted):
                lut[global_id] = local_idx
            args.known_id_lut = lut.to(device)

            args.known_class_names = get_known_class_names(args)
            args.text_emb = build_text_prototypes(args.clip_model, args.known_class_names, device)

        if args.pseudo_align_weight > 0 and os.path.exists(args.pseudo_names_path):
            args.pseudo_proto, args.pseudo_valid = load_pseudo_text_prototypes(
                args.clip_model, args.pseudo_names_path, args.num_species, device)

    train(model, train_loader, test_loader, test_loader_unlabelled, args, get_order_family_target_dict[args.dataset_name], None)
