# ResTune

<p align="center">
    <img src='https://github.com/liuyudut/ResTune/blob/main/images/NCD_ResTune.png' height="250" >
</p>

## Dependencies
- Python 
- PyTorch 
- Numpy 
- scikit-learn 

## Data preparation
Data should be downloaded in `./Datasets/`. You may also use any path by setting the `--dataset_root` argument to `/your/path/`.

## Supervised pre-training with labeled data

```shell
# Train CIFAR-10 with 5 labeled classes
CUDA_VISIBLE_DEVICES=0 python cifar10_pretrain.py 

# Train CIFAR-100 with 80 labeled classes
CUDA_VISIBLE_DEVICES=0 python cifar100_pretrain.py 

# Train TinyImageNet with 80 labeled classes
CUDA_VISIBLE_DEVICES=0 python tinyimagenet_pretrain.py 
```

## Unsupervised clustering with unlabeled data)

```shell
# Train CIFAR-10 with 5 unlabeled classes
CUDA_VISIBLE_DEVICES=0 python cifar10_ResTune.py 

# Train CIFAR-100 with 20 unlabeled classes
CUDA_VISIBLE_DEVICES=0 python cifar100_ResTune.py 

# Train TinyImageNet with 20 unlabeled classes
CUDA_VISIBLE_DEVICES=0 python tinyimagenet_ResTune.py 
```

## Data preparation
The models and results are saved in `./results/`. 
