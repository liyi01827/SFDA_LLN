# SFDA_LLN

## Introduction
PyTorch code for the ICLR 2023 paper [[When Source-Free Domain Adaptation Meets Learning with Noisy Labels](https://openreview.net/pdf?id=u2Pd6x794I)].
```
@inproceedings{
yi2023when,
title={When Source-Free Domain Adaptation Meets Learning with Noisy Labels},
author={Li Yi and Gezheng Xu and Pengcheng Xu and Jiaqi Li and Ruizhi Pu and Charles Ling and Ian McLeod and Boyu Wang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=u2Pd6x794I}
}
```

## Plug-In Example
For those who want to apply our method in their own code, we provide a minimal example:

```python
import torch

train_data = torch.randn(nb_training_samples, 3, 224, 224)

def sample_unlabelled_target_images(train_data):
    index = torch.randint(len(train_data), (batch_size,))
    return index, train_data[index]

class ELR_reg(torch.nn.Module):
    """
    This code implements ELR regularization which is partially adapted from 
    https://github.com/shengliu66/ELR/blob/master/ELR/model/loss.py
    """
    def __init__(self, beta=0.7, lamb=3.0, num, nb_classes):
        super(ELR_loss, self).__init__()
        self.ema = torch.zeros(num, nb_classes).cuda()
        self.beta = beta
        self.lamb = lamb

    def forward(self, index,  outputs):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg
        return final_loss

elr_loss = ELR_reg(num=nb_training_samples, nb_blasses=nb_classes)

for _ in range(100):
    index, images = sample_unlabelled_images()
    # Your model pretrained on source domain
    logits = net(images)

    # Plug-in your an existing (or your own) SFDA loss function 
    loss_sfda = sfda_loss(logits) 
    loss_elr = elr_loss(index, logits)
    loss = loss_sfda + loss_elr
```


## Training Scripts

Please refer to [[run.sh](run.sh)].



## Acknowledgement
This repository inherits some codes from [SHOT](https://github.com/tim-learn/SHOT), [NRC](https://github.com/Albert0147/NRC_SFDA), and [G-SFDA](https://github.com/Albert0147/G-SFDA).
