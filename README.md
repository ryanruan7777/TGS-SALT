# Tgs-Salt-Identification-Chanllenge open Code
We join to [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge). And get the final score 0.860. It's not a good socre but it's our team first time to take part in the Deep learning competition.

## Enviroment
>tensorflow 1.10
keras
numpy
pandas and etc.

## Some attempts
- [ ] CosineAnnealingLR and SWA(not try)
- [x] model selection:VggUnet, traditional Unet, ResNetUnet
- [x] Loss selection:
      BCE+focal_loss worse than the Lovasz_loss
	  we alos combine BCE and dice_loss and iou and etc.But the BCE+Lovasz_loss is the best result.
- [x] 400BCE + 200 Lovasz_loss to train
- [x] ResNet32+SCSE module is better
- [x] K-fold, we set K=5 cause the device limited, and it works, improve 0.015+
- [x] 3 different Depth
- [x] TTA work from 0.808 to 0.814 at single model in 1-fold
### Training Steps
First, we train the resnet32 and without SCSE using the BCE+L_loss and set the 5-fold. And then train the model with the SCSE module 2 times. Last we fuse the first five fold andt the SCSE fold with the TTA get the Public LB 0.843, which in the final score is 0.860. The hyperparameter is in the code.Limited by the time, we could not try many tricks.

## How to start
Train:
- > python main_5folder.py

Test:
- > python test.py


## Some Reference
Thanks for some kaggler share,like Heng and [Peter's discussion](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568).
