# Rethinking-Calibration-of-Deep-Neural-Networks
This is the implementation of our NeurIPS'21 paper (Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence).

Requirements: 
Python 3.6, 
numpy 1.19, 
Pytorch 1.5, 
torchvision 0.6.

You need to:
1. Download CIFAR-10 and CIFAR-100 datasets into './data/'.
2. Run the following demos:
```
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method ce
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method ls --epsilon 0.05

python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method ce
python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method ls --epsilon 0.09
```

The TS algorithm in `temperature_scaling.py` is based on [Geoff Pleiss](https://geoffpleiss.com/)'s [implementation](https://github.com/gpleiss/temperature_scaling).

If you have any further questions, please feel free to send an e-mail to: wangdb@seu.edu.cn. Have fun!

