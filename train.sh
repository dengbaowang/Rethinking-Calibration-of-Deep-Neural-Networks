python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method ce
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method ls --epsilon 0.05
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method l1 --alpha 0.05
python main.py  --dataset cifar10 --seed 101 --batch-size=512 --arch resnet32 --method focal --gamma 3

python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method ce
python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method ls --epsilon 0.09
python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method l1 --alpha 0.01
python main.py  --dataset cifar100 --seed 101 --batch-size=512 --arch resnet32 --method focal --gamma 5
