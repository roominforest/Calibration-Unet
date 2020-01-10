import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
class ce_loss(nn.Module):
    def __init__(self, ignore_index=None,focal = False):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal = focal

    def forward(self, x, label, val=False):
        label = label[:, 0:1]
        if val:
            loss = torch.sum(x.float() * 0)
        else:
            x = x.float()
            log_pos = F.logsigmoid(x).float()
            log_neg = F.logsigmoid(-x).float()

            m_p = label == 1
            m_n = label == 0
            N = torch.sum(label != 255).float()
            if self.focal:
                prob_pos = F.sigmoid(x).float()
                prob_neg = F.sigmoid(x).float()
                l_p = torch.sum(torch.pow(1. - prob_pos[m_p], 2) * log_pos[m_p])
                l_n = torch.sum(torch.pow(1. - prob_neg[m_n], 2) * log_neg[m_n])
            else:
                l_p = torch.sum(log_pos[m_p])
                l_n = torch.sum(log_neg[m_n])

            loss = -1. * (l_p + l_n) / N
        return loss

class Neuron_dice_loss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, x, label):
        # label : two channels
        logit = F.sigmoid(x)
        logit = logit.float()
        label = label[:, 0:1].float()
        vaildmask = label != 255
        vaildmask = vaildmask.float()

        pre = logit * vaildmask
        label = label * vaildmask
        tp = torch.sum(pre * label)
        nOut = torch.sum(pre)
        nLab = torch.sum(label)

        loss1 = -(2. * tp + 1e-5) / (nOut + nLab + 1e-5)
        losses = (loss1.detach())
        total_loss = loss1
        return total_loss, losses


class NeuronDICE(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()

    def forward(self, pred, labels):
        # pred binary type
        labels = labels.float()
        validmask = (labels != 255).float()
        labels = labels * validmask
        tp = torch.sum(pred.float() * labels.float())
        dice = (2. * tp) / (torch.sum(pred).float() + torch.sum(labels).float())

        return dice


class NeuronRecall(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, labels):
        labels = labels[0].float()
        validmask = (labels != 255).float()
        labels = labels.float() * validmask
        tp = torch.sum(pred.float() * labels)
        #         print('tp:',tp)
        N_pos = torch.sum(labels)
        recall = tp.float() / N_pos.float()
        return recall

class NeuronPrecision(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, labels):
        labels = labels[0].float()
        validmask = (labels != 255).float()
        labels = labels.float() * validmask
        tp = torch.sum(pred.float() * labels)
        N_pred = torch.sum(pred.float() * validmask)
        precision = tp.float() / N_pred.float()
        return precision

class Calibration_loss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_loss = Neuron_dice_loss(ignore_index=255)
        self.ce_loss = ce_loss(ignore_index=255,focal = False)

    def forward(self, x, x_s, labels):
        dice, _ = self.dice_loss(x, labels)
        ce = self.ce_loss(x, labels)

        label = labels[:, 0:1].float()
        valid = (label != 255).float()
        label = label.float() * valid

        mask = x.float() * valid
        mask = (mask > 0).float()
        b = mask.size()[0]
        tps = torch.sum((mask*label).reshape([b,-1]),1).float()
        N_ps = torch.sum(mask.reshape([b,-1]),1).float()
        N_ts = torch.sum(label.reshape([b,-1]),1).float()
        target_dice = (2. * tps) / (N_ps + N_ts + 1e-8).float()
        target = target_dice
        s_loss = torch.mean(torch.pow(x_s - target, 2))
        score_average = torch.mean(x_s)
        target_average = torch.mean(target)
        losses = (dice.detach(), s_loss.detach(), ce.detach(), score_average.detach(), target_average.detach())
        loss = 1.5 * s_loss + dice + 0.5 * ce
        return loss, losses