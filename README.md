# Orthogonal-Convolutional-Neural-Networks
[[Project]](http://pwang.pw/ocnn.html) [[Paper]](https://arxiv.org/abs/1911.12207)   

## Overview
This is authors' re-implementation of the orthogonal convolutional neural networks/regularizers described in:  
"[Orthogonal Convolutional Neural Networks](https://arxiv.org/abs/1911.12207)"   
[Jiayun Wang](http://pwang.pw/),&nbsp; [Yubei Chen](https://redwood.berkeley.edu/people/yubei-chen/),&nbsp;  [Rudrasis Chakraborty](https://rudra1988.github.io/),&nbsp; [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/)&nbsp; (UC Berkeley/ICSI)&nbsp; in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2020

For quick addition of the orthogonal loss to your network, refer to [orth_dist](imagenet/utils.py#L42) and [deconv_orth_dist](imagenet/utils.py#L34).

## Updates
As a stand-alone feature, we provide [code](https://github.com/samaonline/CNN-filter-similarity) to calculate filter similarities for CNNs (as described in Fig.1a/1c in the [paper](https://arxiv.org/abs/1911.12207)).

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)

## Overall architecture
This repo will consist of source code of experiments in the paper. Now we released the code for image classification. For classification on your own datasets, just change the folder path and number of classes.

## Image classification

We use imagenet classificaiton as an example. The users can also change the data to CIFAR or other image classification dataset at their interest. The code is heavily based on [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet).

- Navigate to "imagenet" folder.

- We now support orthogonal convolutions for resnet34 and resnet50. You can run resnet50 on imagenet using the following command. 
```
python main_orth50.py --dist-url 'tcp://127.0.0.1:1321' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -a resnet50 -j 4 -r 0.5 -b 220 /data/ILSVRC2012 --print-freq 200
```

- For more details including multi-gpu settings, please refer to [here](imagenet/README.md).

## Note
The current code supports multi-GPU settings.

## Q \& A
Q: What is the difference between "orth_dist" and "deconv_orth_dist" ?

A: As described in our paper, for orthogonalities in fully-connected layers (or certain convolution layers that share the orthogonal condition with fully-connected layers, e.g. kernel size 3x3 and stride 3), we apply "orth_dist". For other convolution layers, we apply "deconv_orth_dist".

Q: Why are specific layers and convs chosen?

A: Empirically we find that for the first conv layer (usually 3 rgb channels to 16/64/... channels), applying the orthogonal loss leads to instable performance. 

## License and Citation
The use of this software is released under [BSD-3](LICENSE).
```
@inproceedings{wang2019orthogonal,
  title={Orthogonal Convolutional Neural Networks},
  author={Wang, Jiayun and Chen, Yubei and Chakraborty, Rudrasis and Yu, Stella X},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
