# Frequency Spatial collaborative remote sensing image super resolution utilizing frequency domain phase and spatial enhancement to reconstruct global structure and texture details


## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- Then run the following codes (taking FSCSR_SRx4_ImageNet-pretrain.pth` as an example):

```
python fscsr/test.py -opt options/test/fscsr_SRx4_ImageNet-pretrain.yml
```

The testing results will be saved in the `./results` folder.  


## How To Train

- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Validation data can be download at [this page](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- The training command is like

```
 python fscsr/train.py -opt options/train/train_fscsr_SRx4_from_scratch.yml 
```

The training logs and weights will be saved in the `./experiments` folder.
