这个代码对 "Fast Deep Stereo with 2D Convolutional Processing of Cost Signatures", WACV, 2020. [FDSCS](https://github.com/ayanc/fdscs) 文章的 `pytorch` 复现。当然，我仅仅是对论文里面的纯 `2d` 卷积结构感兴趣，同时，我想将该代码用在后续的 `FPGA` 工程化部署上。

This code matches "Fast Deep Stereo with 2D Convolutional Processing of Cost Signatures", WACV, 2020. [FDSCS] (https://github.com/ayanc/fdscs) of the article `pytorch` repetition. Of course, I was only interested in the pure '2d' convolution structure in the paper, and I wanted to use the code for subsequent 'FPGA' engineering deployments.

在复现的过程中，我参考了 [fdscs-pytorch](https://github.com/leejiajun/fdscs-pytorch) 和 [GwcNet](https://github.com/xy-guo/GwcNet)的代码实现。

In the process of reproduction, I refer to the [FDSCS - pytorch] (https://github.com/leejiajun/fdscs-pytorch) and code implementation [GwcNet] (https://github.com/xy-guo/GwcNet).

代码在 `sceneflow` 数据集上进行预训练，在 `Kitti2015` 数据集上进行微调。

The code is pre-trained on the 'sceneflow' dataset and fine-tuned on the 'Kitti2015' dataset.

关于 `sceneflow`:

About 'sceneflow' :

我完全借鉴了 `GwcNet` 的训练方式，注意，这可能和原文有一些出入，主要是原文实现的数据集来自 `sceneflow` 一部分，这对我来说需要耗费很多时间去处理。我将 `batch_size`设置为 `4`，`epoch` 设置为 `80`，初始 `5` 个 `epoch` 学习率设置为`10^-4`,后续的使用`10^-5`。论文里面只报告了 `sceneflow` 上的 `EPE` 分数，为`2.01`，这个复现代码跑出来的效果在`2.32`左右，具体的一些指标如下：

I borrowed the GwcNet training completely, note that there may be some differences from the original, mainly the original implementation of the data set from 'sceneflow' part, which took a lot of time for me to process. I set 'batch_size' to '4', 'epoch' to '80', the initial '5' epoch learning rate to '10^-4', and subsequent uses to '10^-5'. The paper only reports the 'EPE' score on 'sceneflow', which is' 2.01 ', and the effect of this repeated code is about '2.32', some specific indicators are as follows:

```txt
avg_test_scalars {'loss': 1.8965044921669272, 'D1': [0.1244640283667954], 'EPE': [2.327345729963871], 'Thres1': [0.40837842607410996], 'Thres2': [0.22138314692804104], 'Thres3': [0.14712449165241592]}
```

关于 `kitti2015`:

About 'kitti2015' :

我将 `batch_size`设置为 `4`，`epoch` 设置为 `5000`，初始 `3000` 个 `epoch` 学习率设置为`10^-4`,`4000`个 epoch 之后使用`10^-5`,`5000`个 `epoch` 之后使用`10^-6`。这个复现代码跑出来具体的一些指标如下：

I set 'batch_size' to '4', 'epoch' to '5000', The learning rate is set to '10^-4' for the initial '3000' epochs, '10^-5' for '4000' epochs, and '10^-6' for '5000' epochs. The specific indicators of this repeated code are as follows:

注意：关于`quant`结尾的代码，你不需要看，这是对 `FPGA` 后续部署所需的一些量化操作，下载代码之后，你只需要运行 `train.py`或者`train_sf.py`来训练 `sceneflow` 或者 `kitti`，代码里面可能有一些中文注释或者错误，请谅解，如果您有更好的复现效果，欢迎一起交流！

Attention: As for the code at the end of 'quant', you don't need to look, it is some quantitative operation required for the subsequent deployment of 'FPGA', after downloading the code, you only need to run 'train.py' or 'train_sf.py' to train 'sceneflow' or 'kitti', There may be some Chinese comments or errors in the code, please understand, if you have a better reproduction effect, welcome to communicate together!

量化的代码：

Quantized code:

`run_quant.py`进行模型量化，并加载测试集进行测试。

'run_quant.py' quantizes the model and loads a test set for testing.
