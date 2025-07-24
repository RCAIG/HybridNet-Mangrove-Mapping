import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp





def getSobel(mask,device):
    SOBEL = nn.Conv2d(1, 2,1,padding=1,bias=False)
    SOBEL.weight.requires_grad = False
    #print(SOBEL.weight)
    SOBEL.weight.set_(torch.Tensor([[[-1, 0, 1], 
                                [-2, 0, 2],
                                [-1, 0, 1]],
                                [[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]]]).reshape(2, 1, 3, 3))
    SOBEL = SOBEL.to(device)

    mask = mask.float()
    with torch.no_grad():
        mask1 = torch.any(SOBEL(mask) != 0, dim=1, keepdims=True).float() 
    return mask1 #B,1,H,W


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



Bceloss = smp.losses.SoftBCEWithLogitsLoss() #One-hot
DiceLoss = smp.losses.DiceLoss(mode = "multilabel")

def criterion(y_pred, y_true, edge_pred, edge_target):
    
    bce_loss = nn.BCELoss(reduction='mean')    
    loss_edge = bce_loss(edge_pred,edge_target)
    
    return  Bceloss(y_pred, y_true)  + DiceLoss(y_pred, y_true)  #+ loss_edge



class evaluate_metric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusionMat = np.zeros((self.num_class, self.num_class), dtype = np.int)
        self.total = 0
        
    def addBatch(self, label, pred):
        label_1d = label.flatten()
        pred_1d = pred.flatten()
        
        for l in range(self.num_class):
            for p in range(self.num_class):
                self.confusionMat[p, l] += np.sum((label_1d == l) & (pred_1d == p))
        
    def getMetrics(self):

        eps = 1e-6
        total_correct_class = np.diag(self.confusionMat)#TP
        total_seen_class = np.sum(self.confusionMat, axis = 0) #TP+FN
        total_pred_class = np.sum(self.confusionMat, axis = 1) #TP + FP
        total_deno_miou_class = total_seen_class + total_pred_class - total_correct_class #(TP+FP+FN)
        miou = np.mean(total_correct_class / (total_deno_miou_class.astype(np.float32) + eps)) #TP / (TP+FN + FP)
        precision = total_correct_class / (total_pred_class.astype(np.float32) + eps) #TP/(TP+FP)
        recall = total_correct_class / (total_seen_class.astype(np.float32) + eps) #TP/(TP+FN)
        F1 = (2.0 * precision * recall) / (precision + recall + eps) # (2*P*R)/(P+R)
        OA = np.sum(total_correct_class) / np.sum(total_seen_class, dtype=np.float32)
        
        return miou, F1, OA