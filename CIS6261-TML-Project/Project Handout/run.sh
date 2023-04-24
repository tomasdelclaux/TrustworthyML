#! /bin/sh

echo "Running CIFAR100 fine ENB0"

python3 part1.py cifar100 fine ENB0 5 2 7

echo "Running CIFAR100 fine ENB4"

python3 part1.py cifar100 fine ENB4 5 2 7

echo "Running CIFAR100 fine DenseNet"

python3 part1.py cifar100 fine DenseNet 5 2 7

echo "Running CIFAR100 fine ResNet"

python3 part1.py cifar100 fine ResNet 5 2 7

mv Results ResultsFine5_2_7

mkdir Results

echo "Running CIFAR100 fine ENB0"

python3 part1.py cifar100 fine ENB0 3 1 7

echo "Running CIFAR100 fine ENB4"

python3 part1.py cifar100 fine ENB4 3 1 7

echo "Running CIFAR100 fine DenseNet"

python3 part1.py cifar100 fine DenseNet 3 1 7

echo "Running CIFAR100 fine ResNet"

python3 part1.py cifar100 fine ResNet 3 1 7
