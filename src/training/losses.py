# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

print("✅ losses.py 불러와짐")

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        num_classes = input.size(1)
        log_probs = F.log_softmax(input, dim=1)

        # 스무딩된 정답 분포 생성
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        focal_term = (1 - pt).pow(self.gamma)
        loss = self.ce(input, target) * focal_term[range(len(target)), target]
        return loss.mean()

print("✅ LabelSmoothingCrossEntropy = ", LabelSmoothingCrossEntropy)