# Rethinking Calibration of Deep Neural Networks: Don't Be Afraid of Overconfidence
This is the implementation of our NeurIPS'21 paper [Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence](https://openreview.net/forum?id=NJS8kp15zzH&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions)). In the paper, we conduct a series of empirical studies showing that overconfidence may not hurt final calibration performance if post-hoc calibration is allowed, rather, the penalty of confident outputs will compress the room of potential improvements in post-hoc calibration phase.

## Dependencies
This code requires the following:

* Python 3.6, 
* numpy 1.19, 
* Pytorch 1.5, 
* torchvision 0.6.

## Training
You need to:

1. Download CIFAR-10 and CIFAR-100 datasets into `./data/`.

2. Run the following demos:
```
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method ce
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method ls --epsilon 0.05

python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method ce
python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method ls --epsilon 0.09
```

## Citation
```
@inproceedings{neurips21dbwang,
author = {Deng-Bao Wang and Lei Feng and Min-Ling Zhang},
title = {Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence},
booktitle = {Advances in Neural Information Processing Systems, Virtual Event},
year = {2021}
}
```

## Acknowledgements
The TS algorithm in `temperature_scaling.py` is based on [Geoff Pleiss](https://geoffpleiss.com/)'s [implementation](https://github.com/gpleiss/temperature_scaling).

## Contact
If you have any further questions, please feel free to send an e-mail to: wangdb@seu.edu.cn.
