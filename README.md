# self-attention-GAN-pytorch

This is an almost exact replica in PyTorch of the Tensorflow version of [SAGAN](https://arxiv.org/abs/1805.08318) released by Google Brain [[repo](https://github.com/brain-research/self-attention-gan)] in August 2018.

Code structure is inspired from [repo](https://github.com/heykeetae/Self-Attention-GAN), but follows the details of Google Brain's [repo](https://github.com/brain-research/self-attention-gan).

# Self-Attention GAN
**[Han Zhang, Ian Goodfellow, Dimitris Metaxas and Augustus Odena, "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)](https://arxiv.org/abs/1805.08318).**

## Prerequisites

Check `requirements.txt`.

* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.1](http://pytorch.org/)

&nbsp;

## Usage

#### 1. Check `parameters.py` for all arguments and their default values

#### 2. Train on custom images in folder a/b/c:
```bash
$ python train.py --data_path 'a/b/c' --save_path 'o/p/q' --batch_size 64 --name sagan
```

(Warning: Works only on *128x128* images, input images are resized to that. Tweak the Generator first if you would like to use some other image size. And then use `imsize` option:
```bash
$ python train.py --data_path 'a/b/c' --save_path 'o/p/q' --batch_size 64 --imsize 64 --name sagan
```
)

Model training will be recorded in a new folder inside `--save_path` with the name `<timestamp>_<name>_<basename of data_path>`.

By default, model weights are saved in a subfolder called `weights`, and train & validation samples during training in a subfolder called `samples` (can be changed in `parameters.py`).

@article{Zhang2018SelfAttentionGA,
    title={Self-Attention Generative Adversarial Networks},
    author={Han Zhang and Ian J. Goodfellow and Dimitris N. Metaxas and Augustus Odena},
    journal={CoRR},
    year={2018},
    volume={abs/1805.08318}
}
