# ShRank (Shrink Rank)
### Automatic factorization library for pytorch

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Forked from [Greenformer](https://github.com/SamuelCahyawijaya/greenformer), the official implementation of the paper:
- Winata, G. I., Cahyawijaya, S., Lin, Z., Liu, Z., & Fung, P. (2020, May). Lightweight and efficient end-to-end speech recognition using low-rank transformer. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6144-6148). IEEE.

### What is ShRank?
ShRank is a library to convert `Linear`, GPT-style "`Conv1D`" (really just transposed Linear), `Conv1d` (torch, actual 1D convolution), `Conv2d`, `Conv3d` layers into its own variants called `LED`/`CED`.
ShRank seeks over your PyTorch model and replaces all `Linear`/GPT "`Conv1D`" layers into `LED` modules and all `Conv1d`, `Conv2d`, `Conv3d` modules into `CED` layers with the specified rank (as long as it makes sense to approximate them, i.e. doing so would actually trim parameters).

For the time being, ShRank is focused exclusively on using a faster, non-random implementation of SVD (`torch.linalg.svg` instead of `torch.svd_lowrank` used in Greenformer's SVD solver). Removed Greenformer's NMF and SNMF solvers (due to unmaintained dependency AFAICT), as well as the random (from scratch) solver (just do `from shrank.lr_module import LED, CED` and use them in place of [`nn.Linear`, `transformers.modeling_utils.Conv1D` (GPT-style '1D convolution' aka transposed linear)] or [`nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`], respectively).

### How to Install
```
# Not yet, will consider publishing to PyPI
# pip install shrank

# Use this instead for now
pip install git+https://github.com/ScienceArtMagic/ShRank.git
```

### Usage (TODO: update)

The examples below only convert the PyTorch model (in memory).

Save it according to the docs of Hugging Face Transformers, PyTorch, TorchVision, etc.

At minimum, you will need to add `rank` to your model configuration (as well as `groups_out` for convolutions, if you enable this during conversion). You'll also need to replace references to the original linear or convolution modules with LED or CED, respectively, in your e.g. `modeling_{model}.py`

#### Grouped Convolutions

In the case of grouped convolutions where the `groups` value is larger than the specified rank (e.g. Mamba), `auto_fact` will skip them by default (`skip_high_groups=True`). You can override this by setting `skip_high_groups=False` (depending on your rank, this might ***add*** parameters, counterintuitively). Since `groups` can't be higher than a convolution's `in_channels` or `out_channels` (and `ced_module[0]`'s `out_channels` value is the rank), this will change its `groups` value to rank.

By default, only the first low-rank layer `ced_module[0]` sets groups (the lesser of rank or the `groups` of the original conv module). If you want `ced_module[1]` to be grouped as well (same as `ced_module[0]`, you can set `groups_out=True`).

##### BERT Model
```
from transformers import BertModel, BertConfig
from shrank import auto_fact

config = BertConfig.from_pretrained('bert-base-uncased', pretrained=False)
model = BertModel(config=config)

model = auto_fact(model, rank=100, deepcopy=False, fact_led_unit=False)
```

##### VGG Model
```
import torch
from torchvision import models
from shrank import auto_fact

model = models.vgg16()
model = auto_fact(model, rank=64, deepcopy=False, fact_led_unit=False)
```

### Why Use ShRank (WIP/TODO: update notebooks)
- Improve the speed of you model significantly, check [Example Notebook](https://github.com/SamuelCahyawijaya/py_auto_fact/blob/main/examples/factorize_bert.ipynb)
- Maintain model performance with appropriate choice of rank, check our [ICASSP 2020 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053878)
- Easy to use and works on any kind of model!
