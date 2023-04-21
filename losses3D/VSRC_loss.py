import numpy as np
import torch
from torch import nn
import math
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch.nn.functional as F


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]


def mse_Loss(probs, targets, reduction='mean'):
    if reduction == 'mean':
        criteria = nn.MSELoss(reduction='mean')
    elif reduction == 'sum':
        criteria = nn.MSELoss(reduction='sum')
    else:
        criteria = nn.MSELoss(reduction='none')
    mse_loss = criteria(probs, targets)
    return mse_loss


def dice_loss(score, target):
    target = target.to(torch.float32)
    # target = target.float()
    smooth = 1e-10
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def compute_sdf(img_gt, out_shape):
    """
    Semi-supervised Medical Image Segmentation through Dual-task Consistency
    https://ojs.aaai.org/index.php/AAAI/article/view/17066

    code refer to:
    https://github.com/HiLab-git/DTC/blob/master/code/utils/util.py

    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                    np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf

    return normalized_sdf

class DTU_Student_Loss():
    def __init__(self, args, targets, l_output, r_output, l_output_sdf, r_output_sdf, le_output=None, re_output=None,
                 le_output_sdf=None, re_output_sdf=None, epc=0):
        self.args = args
        self.epc = epc * 0.5
        # self.epc = epc
        self.targets = targets
        self.l_output = l_output
        self.r_output = r_output
        self.l_output_sdf = l_output_sdf
        self.r_output_sdf = r_output_sdf
        self.le_output = le_output
        self.re_output = re_output
        self.le_output_sdf = le_output_sdf
        self.re_output_sdf = re_output_sdf
        self.save_fig_path = './save_figure/'


    def cal_train_loss(self):

        l_output_labeled = self.l_output[:self.args.labeled_bs]
        r_output_labeled = self.r_output[:self.args.labeled_bs]
        l_output_labeled_sdf = self.l_output_sdf[:self.args.labeled_bs]
        r_output_labeled_sdf = self.r_output_sdf[:self.args.labeled_bs]

        l_output_unlabeled = self.l_output[self.args.labeled_bs:]
        r_output_unlabeled = self.r_output[self.args.labeled_bs:]
        le_output_unlabeled = self.le_output[self.args.labeled_bs:]
        re_output_unlabeled = self.re_output[self.args.labeled_bs:]

        target_label = self.targets

        # Segmentation Loss
        l_output_labeled_soft = F.softmax(l_output_labeled, dim=1)
        r_output_labeled_soft = F.softmax(r_output_labeled, dim=1)
        l_dice = 0.0
        r_dice = 0.0

        l_loss_dice = dice_loss(l_output_labeled_soft[:, 0, :, :, :], target_label)
        r_loss_dice = dice_loss(r_output_labeled_soft[:, 0, :, :, :], target_label)
        l_dice += l_loss_dice
        r_dice += r_loss_dice

        # SDF Loss
        b, c, d, w, h = l_output_labeled.size()
        sdf = compute_sdf(target_label.detach().cpu().numpy(), (b, d, w, h))
        sdf = torch.from_numpy(sdf).cuda().float()
        l_output_labeled_sdf_c0 = l_output_labeled_sdf[:, 0, ...]
        r_output_labeled_sdf_c0 = r_output_labeled_sdf[:, 0, ...]
        l_sdfLoss = mse_Loss(l_output_labeled_sdf_c0, sdf)
        r_sdfLoss = mse_Loss(r_output_labeled_sdf_c0, sdf)

        # Consistency Loss (uncertainty) Seg
        consistency_weight = 0.1 * math.exp(-5 * math.pow((1 - self.epc / self.args.nEpochs), 2))
        l_output_soft = F.softmax(self.l_output, dim=1)
        le_output_soft = F.softmax(self.le_output, dim=1)
        r_output_soft = F.softmax(self.r_output, dim=1)
        re_output_soft = F.softmax(self.re_output, dim=1)

        l_p_seg_bool = (l_output_soft > self.args.tau_p) * (le_output_soft > self.args.tau_p)
        l_n_seg_bool = (l_output_soft <= self.args.tau_n) * (le_output_soft <= self.args.tau_n)
        l_certain_seg_bool = l_p_seg_bool + l_n_seg_bool

        r_p_seg_bool = (r_output_soft > self.args.tau_p) * (re_output_soft > self.args.tau_p)
        r_n_seg_bool = (r_output_soft < self.args.tau_n) * (re_output_soft < self.args.tau_n)

        r_certain_seg_bool = r_p_seg_bool + r_n_seg_bool

        l_certain_seg = l_output_soft * (l_certain_seg_bool == True)
        le_certain_seg = le_output_soft * (l_certain_seg_bool == True)
        r_certain_seg = r_output_soft * (r_certain_seg_bool == True)
        re_certain_seg = re_output_soft * (r_certain_seg_bool == True)
        l_uncertain_seg = l_output_soft * (l_certain_seg_bool == False)
        le_uncertain_seg = le_output_soft * (l_certain_seg_bool == False)
        r_uncertain_seg = r_output_soft * (r_certain_seg_bool == False)
        re_uncertain_seg = re_output_soft * (r_certain_seg_bool == False)

        # reliable consLoss seg
        lc_consLoss_seg = torch.mean(torch.pow(l_certain_seg - le_certain_seg, 2))
        rc_consLoss_seg = torch.mean(torch.pow(r_certain_seg - re_certain_seg, 2))

        # unreliable consLoss seg
        le_uncertainty = -1.0 * torch.sum(le_uncertain_seg * torch.log2(le_uncertain_seg + 1e-6), dim=1, keepdim=True)
        re_uncertainty = -1.0 * torch.sum(re_uncertain_seg * torch.log2(re_uncertain_seg + 1e-6), dim=1, keepdim=True)

        luc_consLoss_seg = torch.mean(torch.pow(l_uncertain_seg - le_uncertain_seg, 2) * torch.exp(-le_uncertainty))
        ruc_consLoss_seg = torch.mean(torch.pow(r_uncertain_seg - re_uncertain_seg, 2) * torch.exp(-re_uncertainty))

        # Consistency Loss (uncertainty) SDF
        l_certain_sdf_bool = [(self.l_output_sdf > self.args.sdf_threshold) * (self.le_output_sdf > self.args.sdf_threshold)]\
                             + [(self.l_output_sdf < -self.args.sdf_threshold) * (self.le_output_sdf < -self.args.sdf_threshold)]
        r_certain_sdf_bool = [(self.r_output_sdf > self.args.sdf_threshold) * (self.re_output_sdf > self.args.sdf_threshold)]\
                             + [(self.r_output_sdf < -self.args.sdf_threshold) * (self.re_output_sdf < -self.args.sdf_threshold)]

        # reliable consLoss sdf
        l_certain_sdf = self.l_output_sdf * (l_certain_sdf_bool == True)
        le_certain_sdf = self.le_output_sdf * (l_certain_sdf_bool == True)
        r_certain_sdf = self.r_output_sdf * (r_certain_sdf_bool == True)
        re_certain_sdf = self.re_output_sdf * (r_certain_sdf_bool == True)

        lc_consLoss_sdf = torch.mean(torch.pow(l_certain_sdf - le_certain_sdf, 2))
        rc_consLoss_sdf = torch.mean(torch.pow(r_certain_sdf - re_certain_sdf, 2))

        # unreliable consLoss sdf
        l_uncertain_sdf = self.l_output_sdf * (l_certain_sdf_bool == False)
        le_uncertain_sdf = self.le_output_sdf * (l_certain_sdf_bool == False)
        r_uncertain_sdf = self.r_output_sdf * (r_certain_sdf_bool == False)
        re_uncertain_sdf = self.re_output_sdf * (r_certain_sdf_bool == False)

        le_uncertainty_sdf = le_uncertain_sdf.var()
        re_uncertainty_sdf = re_uncertain_sdf.var()
        luc_consLoss_sdf = torch.mean(torch.pow(l_uncertain_sdf - le_uncertain_sdf, 2) * torch.exp(-le_uncertainty_sdf))
        ruc_consLoss_sdf = torch.mean(torch.pow(r_uncertain_sdf - re_uncertain_sdf, 2) * torch.exp(-re_uncertainty_sdf))

        # Dual-task Cons Loss
        l_output_sdf_c0 = self.l_output_sdf[:, 0, ...]
        r_output_sdf_c0 = self.r_output_sdf[:, 0, ...]
        l_output_c0 = self.l_output[:, 0, ...]
        r_output_c0 = self.r_output[:, 0, ...]
        l_seg_2sigmoid = torch.sigmoid(-2.0 * l_output_c0)
        r_seg_2sigmoid = torch.sigmoid(-2.0 * r_output_c0)

        l_dual_task_consistency = torch.mean(torch.pow((2 * l_seg_2sigmoid - 1) - l_output_sdf_c0, 2))
        r_dual_task_consistency = torch.mean(torch.pow((2 * r_seg_2sigmoid - 1) - r_output_sdf_c0, 2))

        # Stabilization Loss
        l_output_unlabeled_soft = F.softmax(l_output_unlabeled, dim=1)
        r_output_unlabeled_soft = F.softmax(r_output_unlabeled, dim=1)
        le_output_unlabeled_soft = F.softmax(le_output_unlabeled, dim=1)
        re_output_unlabeled_soft = F.softmax(re_output_unlabeled, dim=1)

        l_stable = mse_Loss(l_output_unlabeled_soft, le_output_unlabeled_soft, reduction='none')
        r_stable = mse_Loss(r_output_unlabeled_soft, re_output_unlabeled_soft, reduction='none')

        l_stable_bool = (l_output_unlabeled_soft > self.args.label_threshold) * \
                        (le_output_unlabeled_soft > self.args.label_threshold) * (l_stable < self.args.stable_threshold)
        r_stable_bool = (r_output_unlabeled_soft > self.args.label_threshold) * \
                        (re_output_unlabeled_soft > self.args.label_threshold) * (r_stable < self.args.stable_threshold)

        l_stable_bool_np = l_stable_bool.detach().cpu().numpy()
        r_stable_bool_np = r_stable_bool.detach().cpu().numpy()
        l_stable_np = l_stable.detach().cpu().numpy()
        r_stable_np = r_stable.detach().cpu().numpy()
        l_output_unlabeled_np = l_output_unlabeled_soft.detach().cpu().numpy()
        r_output_unlabeled_np = r_output_unlabeled_soft.detach().cpu().numpy()

        # a1 = np.where(n1 * n2, np.where(s1 < s2, p1, p2), np.where(n1, p1, np.where(n2, p2, p1)))

        l_stabled_output_np = np.where(l_stable_bool_np * r_stable_bool_np,
                                       np.where(l_stable_np < r_stable_np, l_output_unlabeled_np, r_output_unlabeled_np),
                                       np.where(l_stable_bool_np, l_output_unlabeled_np,
                                                np.where(r_stable_bool_np, r_output_unlabeled_np, l_output_unlabeled_np)))
        r_stabled_output_np = np.where(r_stable_bool_np * l_stable_bool_np,
                                       np.where(r_stable_np < l_stable_np, r_output_unlabeled_np, l_output_unlabeled_np),
                                       np.where(r_stable_bool_np, l_output_unlabeled_np,
                                                np.where(l_stable_bool_np, l_output_unlabeled_np, r_output_unlabeled_np)))
        l_stabled_output = torch.from_numpy(l_stabled_output_np).cuda()
        r_stabled_output = torch.from_numpy(r_stabled_output_np).cuda()

        stabilization_weight = 0.1 * math.exp(-5 * math.pow((1 - self.epc / self.args.nEpochs), 2))
        l_stabilization_loss = mse_Loss(l_stabled_output, l_output_unlabeled_soft)
        r_stabilization_loss = mse_Loss(r_stabled_output, r_output_unlabeled_soft)

        l_consLoss = (lc_consLoss_seg + luc_consLoss_seg) + (lc_consLoss_sdf + luc_consLoss_sdf) + l_dual_task_consistency
        r_consLoss = (rc_consLoss_seg + ruc_consLoss_seg) + (rc_consLoss_sdf + ruc_consLoss_sdf) + r_dual_task_consistency

        l_loss = (l_dice + l_sdfLoss) + consistency_weight * l_consLoss + stabilization_weight * l_stabilization_loss
        r_loss = (r_dice + r_sdfLoss) + consistency_weight * r_consLoss + stabilization_weight * r_stabilization_loss

        return l_loss, r_loss, l_dice, r_dice

    def cal_val_loss(self):
        target_label = self.targets
        l_output_labeled_soft = F.softmax(self.l_output, dim=1)
        r_output_labeled_soft = F.softmax(self.r_output, dim=1)
        l_dice = 0.0
        r_dice = 0.0
        l_loss_dice = dice_loss(l_output_labeled_soft[:, 0, :, :, :], target_label)
        r_loss_dice = dice_loss(r_output_labeled_soft[:, 0, :, :, :], target_label)
        l_dice += l_loss_dice
        r_dice += r_loss_dice

        return l_dice, r_dice

