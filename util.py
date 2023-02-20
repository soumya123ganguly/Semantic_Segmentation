import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    #pred_mask = torch.argmax(pred, 1)
    #iou = torch.zeros((len(pred_mask),))
    #num = torch.zeros((len(pred_mask),))
    #for i in range(n_classes):
    #  tp_logic = torch.logical_and(pred_mask == target, pred_mask == i)
    #  tp_mask = torch.where(tp_logic, 1, 0).sum(1).sum(1)
    #  fp_mask = torch.where(pred_mask == i, 1, 0).sum(1).sum(1)
    #  fn_mask = torch.where(target == i, 1, 0).sum(1).sum(1)
    #  ioui = torch.where(tp_mask == 0, 0, (tp_mask)/(fn_mask+fp_mask-tp_mask))
    #  iou = ioui+iou
    #  numi = torch.where(tp_mask == 0, 0, 1)
    #  num = numi+num
    #score = torch.where(num == 0, 0, iou/num).mean()
    pred_mask = torch.argmax(pred, 1)
    score = 0.0
    num = 0
    for i in range(n_classes):
      tp_mask = torch.logical_and(pred_mask == target, pred_mask == i)
      tp = torch.where(tp_mask, 1, 0).sum(dtype=torch.float)
      fp = torch.where(pred_mask == i, 1, 0).sum(dtype=torch.float)
      fn = torch.where(target == i, 1, 0).sum(dtype=torch.float)
      if fp+fn-tp != 0:
        score += (tp/(fp+fn-tp))
        num += 1
    return (score/num)

    #pred_mask = torch.argmax(pred, 1)
    #score = 0.0
    #eps = 1e-2
    #for i in range(n_classes):
    #  tp_logic = torch.logical_and(pred_mask == target, pred_mask == i)
    #  tp_mask = torch.where(tp_logic, 1, 0).sum()
    #  fp_mask = torch.where(pred_mask == i, 1, 0).sum()
    #  fn_mask = torch.where(target == i, 1, 0).sum()
    #  score += (tp_mask+eps)/(fp_mask+fn_mask-tp_mask+eps)
    #score /= (n_classes)
    #return score

def pixel_acc(pred, target):
    pred_mask = torch.argmax(pred, 1)
    bin_mask = torch.where(pred_mask == target, 1, 0).mean(dtype=torch.float)
    return bin_mask