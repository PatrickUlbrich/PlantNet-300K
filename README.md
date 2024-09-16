# PlantNet-300K

<p align="middle">
  <img src="/images/1.jpg" width="180" hspace="2"/>
  <img src="/images/2.jpg" width="180" hspace="2"/>
  <img src="/images/3.jpg" width="180" hspace="2"/>
  <img src="/images/4.jpg" width="180" hspace="2"/>
</p>

This repository is a fork of the original PlantNet-300k repository that was used to produce the benchmark in the paper *"Pl@ntNet-300K: a plant image dataset with high label
ambiguity and a long-tailed distribution"*.

The publicly available code and dataset was used in a study project as part of my masters degree in Data Science. my final examination paper can be found in the file *examination_plantnet_300k_optimizations_MoE.pdf*. The Jupyter Notebook *plantnet_300k_ensemble_MoE* contains most of the additional code I created. I also implemented some changes in the files *cli.py*, *utils.py* and *main.py*. All changes are commented as "*# start of my additional code*" and end with "*# end of my additional code*". Additional information can also be found in the original repository.

## Download the dataset

In order to train a model on the PlantNet-300K dataset, you first have to [download the dataset on Zenodo](https://zenodo.org/record/5645731#.Yuehg3ZBxPY).

## Pre-trained models

You can find the pre-trained models [here](https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/).
To load the pre-trained models, you can simply use the *load_model* function in *utils.py*. For instance, if you want to load the resnet18 weights:

```python
from utils import load_model
from torchvision.models import resnet18

filename = 'resnet18_weights_best_acc.tar' # pre-trained model path
use_gpu = True  # load weights on the gpu
model = resnet18(num_classes=1081) # 1081 classes in Pl@ntNet-300K

load_model(model, filename=filename, use_gpu=use_gpu)
```

Note that if you want to fine-tune the model on another dataset, you have to change the last layer. You can find examples in the *get_model* function in *utils.py*. 

## Requirements

Only pytorch, torchvision are necessary for the code to run. 
If you have installed anaconda, you can run the following command:

```conda env create -f plantnet_300k_env.yml```

## Training a model

In order to train a model on the PlantNet-300K dataset, run the following command:

```python main.py --lr=0.01 --batch_size=32 --mu=0.0001 --n_epochs=30 --epoch_decay 20 25 --k 1 3 5 10 --model=resnet18 --pretrained --seed=4 --image_size=256 --crop_size=224 --root=path_to_data --save_name_xp=xp1```

 You must provide in the "root" option the path to the train val and test folders. 
 The "save_name_xp" option is the name of the directory where the weights of the model and the results (metrics) will be stored.
 You can check out the different options in the file cli.py.
