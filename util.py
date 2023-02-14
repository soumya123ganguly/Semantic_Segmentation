import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    pred_mask = torch.argmax(pred, 1)
    score = 0
    eps = 1e-2
    for i in range(n_classes):
      tp_logic = torch.logical_and(pred_mask == target, pred_mask == i)
      tp_mask = torch.where(tp_logic, 1, 0).sum(dtype=torch.float)
      fp_logic = torch.logical_and(pred_mask != target, pred_mask == i)
      fp_mask = torch.where(fp_logic, 1, 0).sum(dtype=torch.float)
      fn_logic = torch.logical_and(pred_mask != target, target == i)
      fn_mask = torch.where(fn_logic, 1, 0).sum(dtype=torch.float)
      score += (tp_mask+eps)/(fn_mask+fp_mask+tp_mask+eps)
    score /= (n_classes)
    return score

def pixel_acc(pred, target):
    pred_mask = torch.argmax(pred, 1)
    bin_mask = torch.where(pred_mask == target, 1, 0).mean(dtype=torch.float)
    return bin_mask