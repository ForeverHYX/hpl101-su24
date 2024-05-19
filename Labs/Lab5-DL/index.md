# 实验五：简单神经网络训练与加速

## 1 实验简介

**深度学习**（Deep Learning）是[机器学习](https://zh.wikipedia.org/wiki/机器学习)的分支，是一种以[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络)为架构，对数据进行表征学习的[算法](https://zh.wikipedia.org/wiki/算法)。深度学习能够取得如此卓越的成就，除了优越的算法、充足的数据，更离不开强劲的算力。近年来，深度学习相关的基础设施逐渐成熟，从网络设计时的训练、优化，到落地的推理加速，都有非常优秀的解决方案。其中，对于算力的需求最大的部分之一是网络的训练过程，它也因此成为 HPC 领域经常研究的话题。

**卷积神经网络**（Convolutional Neural Network, **CNN**）是一种[前馈神经网络](https://zh.wikipedia.org/wiki/前馈神经网络)，对于大型图像处理有出色表现。

本次实验我们将完成 LeNet-5 的训练，并尝试编写自定义算子。


## 2 实验环境

同 Lab3 和 Lab4。使用 A100 分区的 GPU08 节点，上有两张 A100 80G 显卡。

对于非 Bonus 部分，GPU06 节点的 2080Ti 也可以满足实验要求，你也可以使用该节点进行调试等工作。

集群上安装了一个基础的 torch 环境，使用 conda 管理，你可以直接使用这个环境
```bash
conda activate torch
```
或者自己新建一个环境。比如克隆基础 torch 环境
```bash
conda create -n mytorch --clone torch
```
或者新建一个 python 版本为 3.10 的空环境
```bash
conda create -n mytorch python=3.10
```
注意基础 torch 环境无法被修改（安装新包），因此若你需要安装新包，需要新建一个环境，之后使用 conda 或者 pip 进行管理。

新建的环境会被存放在 `~/.conda/envs/` 目录下，你可以使用 `conda env list` 查看当前环境列表，使用 `conda activate mytorch` 切换到你的环境。

当你混合使用 conda 和 pip 时，注意 pip 是否为当前环境下的 pip，可以使用 `which pip` 查看。

你可以使用 `nvidia-smi` 查看显卡占用情况，设置 `CUDA_VISIBLE_DEVICES` 环境变量来指定使用哪张显卡。例如

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```
指定使用编号为 1 的显卡进行训练，此时只有编号为 1 的显卡对于你的程序可见。

## 3 实验基础知识介绍

### 3.1 网络模型

#### 3.1.1 CNN 卷积神经网络

卷积神经网络由一个或多个卷积层和顶端的全连通层（对应经典的神经网络）组成，同时也包括关联权重和池化层（pooling layer）。这一结构使得卷积神经网络能够利用输入数据的二维结构。与其他深度学习结构相比，卷积神经网络在图像和[语音识别](https://zh.wikipedia.org/wiki/语音识别)方面能够给出更好的结果。这一模型也可以使用[反向传播算法](https://zh.wikipedia.org/wiki/反向传播算法)进行训练。相比较其他深度、前馈神经网络，卷积神经网络需要考量的参数更少，使之成为一种颇具吸引力的深度学习结构。

#### 3.1.2 LeNet-5

LeNet-5是一个较简单的卷积神经网络。下图显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后输出每种分类预测得到的概率。

![](index.assets/LeNet.jpg)

有关于其更详细的结构可以在[原论文](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)中找到。

### 3.2 数据集

#### 3.2.1 MNIST 手写数字数据集

MNIST 数据集 (Mixed National Institute of Standards and Technology database) 是美国国家标准与技术研究院收集整理的大型手写数字数据库，包含 60,000 个示例的训练集以及 10,000 个示例的测试集。

<img src="index.assets/MNIST.jpeg" alt="How to Train a Model with MNIST dataset | by Abdullah Furkan Özbek | Medium" style="zoom:50%;" />

MNIST 数据集下载：http://yann.lecun.com/exdb/mnist/index.html

## 4 实验步骤

### 4.1 LeNet-5 训练

#### 4.1.1 数据准备

我们建议利用 `torchvision` 提供的 `torchvision.datasets` 方法导入数据，`torchvision.datasets` 所提供的接口十分方便，之后你可以用 `torch.utils.data.DataLoader` 给你的模型加载数据。

此外，我们也欢迎你自定义你的 `Dataset` 类，这样做会给你带来额外的分数。为此，你需要继承 `torch.utils.data.Dataset` 并至少需要重写其中的 `__len__()` 和 `__getitem__()` 函数，[这里](https://pytorch.org/docs/stable/data.html)有官方对 `torch.utils.data` 类的介绍，它或许可以帮到你。

幸运的是，本次实验需要用到的 `MNIST` 数据集可用 `torchvision.datasets` 导入，下面对一些你可能会用到的参数简单加以说明

**注意：请在清楚参数含义后调用它们**

```Python
# MNIST
torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```

一些重要的参数说明：

- root: 在 `MNIST`中是 `processed/training.pt` 和 `processed/test.pt` 的主目录
- train: `True` 代表训练集，`False` 代表测试集
- transform 和 target_transform: 分别是对图像和 label 的转换操作
- download: 若为 `True` 则下载数据集并放到 `root` 所指定的目录中，否则直接尝试从 `root` 目录中读取

你可以在[这里](https://pytorch.org/vision/0.8/datasets.html)获取更加详细的说明

#### 4.1.2 模型编写

##### 4.1.2.1 网络结构

`PyTorch` 提供了许多种定义模型的方式，最常用的一种是将网络结构以类保存，你应当首先继承 [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)，并实现正向传播的 `forward` 函数，(为什么不用定义反向传播函数呢？因为你继承的 `nn.Module` 就是干这个事情的)。

下面为网络结构的一个 sample（但显然这样的网络并不能用于本次 Lab），本次实验中你需要自定义你的网络结构，以完成我们的分类任务：

```Python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # 利用参数初始化父类
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

当然，你需要实例化你的模型，可以直接对模型打印以查看结构

```Python
model = Model()
print(model)
```

网络结构编写中一个很大的难点在于每一步的 tensor shape 需要匹配，请仔细检查你的代码来确保此部分的正确性。

##### 4.1.2.2 损失函数

常见的损失函数都被定义在了 `torch.nn`中，你可以在训练过程开始前将其实例化，并在训练时调用，例如：

```Python
criterion = torch.nn.CrossEntropyLoss()
```

##### 4.1.2.3 正向传播

正向传播是指对神经网络沿着从输入层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）。
正向传播的过程在 `forward`中定义，对于模型实例，可以直接利用输入输出得到模型预测的结果。

```Python
y_pred = model(x)
```

##### 4.1.2.4 反向传播

反向传播（Backpropagation，BP）是“误差反向传播”的简称，是一种与最优化方法（如梯度下降法）结合使用的，用来训练人工神经网络的常见方法。该方法对网络中所有权重计算损失函数的梯度。这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数。

在计算过模型的loss之后，可以利用 `loss.backward()` 计算反向传播的梯度，梯度会被直接储存在 `requires_grad=True` 的节点中，不过此时节点的权重暂时不会更新，因此可以做到梯度的累加。

##### 4.1.2.5 优化器

常用的优化器都被定义在了 `torch.optim` 中，为了使用优化器，你需要构建一个 optimizer 对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。你需要给它一个包含了需要优化的参数（必须都是 Variable 对象）的iterable。然后，你可以设置optimizer的参数选项，比如学习率，权重衰减，例如：

```Python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

所有的optimizer都实现了step()方法，这个方法会更新所有的参数。或许你会在反向传播后用到它。

```Python
optimizer.step()
```

需要注意的是，在反向传播前，如果你不希望梯度累加，请使用下面的代码将梯度清零。

```Python
optimizer.zero_grad()
```

#### 4.1.3 训练过程

前文中已经定义了网络结构、损失函数、优化器，至此，一个较为完整的训练过程如下，需要注意的是，你的训练过程要不断从 `DataLoader` 中取出数据。

```Python
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### 4.1.4 测试过程
一般来说，神经网络会多次在训练集上进行训练，一次训练称之为一个 epoch。每个 epoch 结束后，我们会在测试集上进行测试，以评估模型的性能。在测试过程中，我们不需要计算梯度也不可以计算梯度（思考为什么），此时可以使用 `torch.no_grad` 来实现这一点。


```Python
with torch.no_grad():
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
```

#### 4.1.4 TensorBoard

TensorBoard 是常用的训练过程可视化工具。请参考 [PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 的官方教程完成配置。

#### 4.1.5 Tips

- `nn.functional.relu`  （简记为 `F.relu` ）和 `nn.ReLU` 略有不同，区别在于前者作为一个函数调用，而后者作为一个层结构，必须添加到 `nn.Module` 容器中才能使用，两者实现的功能一样，在 `PyTorch` 中，`nn.X` 都有对应的函数版本 `F.X`。
- 除了利用继承 `nn.Module` 来建立网络，不推荐但可以使用 `nn.ModuleList`, `nn.ModuleDict`，推荐使用 `nn.Sequential`直接定义模型
- 你可以定义如下的 `device` 变量，以便你的模型在没有 GPU 环境下也可以测试：

    ```Python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)
    some_data = some_data.to(device)
    ```

- 你不必严格按照原版 LeNet-5 的网络结构来实现，包括超参数、优化器的选择不限，但是你需要保证你的网络结构是合理的，且能够完成我们的分类任务，最终的测试集准确率需要达到 98% 以上。（实际上原版 LeNet 可以轻松达到这个准确率，使用更加现代的结构和优化器，你可以达到更高的准确率）
- 不必过度关注准确率和 loss，只要你的模型能够正常训练，你就可以通过本次实验。

<!-- - 相比于原生的 `PyTorch`，`PyTorch Lightning` 框架对其进行了更高层次的封装，很大程度上简化了模型定义、训练以及测试的步骤，使用 `PyTorch Lightning` 作为本次实验的加分项，官网链接已附在参考资料中。如果你能够在 TensorBoard 中将中间层可视化，你能得到更多的加分。 -->

### 4.2 自定义算子

#### 4.2.1 算子编写
除了 `torch.nn` 中提供的算子，你可以自定义带有自动微分功能的算子。为了简单起见，本实验中你需要编写一个 `GELU` 算子。你需要继承 `torch.autograd.Function` 并实现 `forward` 和 `backward` 函数。`forward` 函数接受输入并返回输出，`backward` 函数接受输入和梯度，并返回相对于输入的梯度。你可以在[这里](https://pytorch.org/docs/stable/notes/extending.html)找到更多关于自定义算子的信息。

关于 `GELU` 本身的信息请从参考资料中查找。你可以自行选择使用近似或者非近似的版本。

编写完自动微分算子后，你需要验证其正确性，可以直接和 `torch` 中的实现进行对比。

```python
import torch.nn.functional as F
loss_func = F.mse_loss()
A = torch.randn(100)
B = A.clone()
A.requires_grad = True
B.requires_grad = True
c = torch.randn(100)
a = F.gelu(A)
b = my_gelu(B)
loss1 = loss_func(a, c)
loss2 = loss_func(b, c) # loss1 should equal to loss2
loss1.backward()
loss2.backward()
gradA = A.grad
gradB = B.grad
err = loss_func(gradA, gradB) # err should equal to 0
```


#### 4.2.2 使用 C++ 

如果你尝试过寻找 `F.gelu` 的实现，你会发现它并不在 PyTorch 的 python 源码里。实际上它是用 C++/CUDA 实现的，你可以尝试寻找它的源码。这里你需要参照[这里](https://pytorch.org/tutorials/advanced/cpp_extension.html#jit-compiling-extensions)的教程，将你的算子用 C++ 重写，并在 python 中调用。

为了简单起见，你可以直接使用 torch 提供的 C++ tensor 数学函数（比如 `exp`）。

之后和上一小节相同，验证你实现的正确性。由于 GELU 本身比较简单，这里不涉及到使用 CUDA 编写算子。

最后，将你编写的算子用于 LeNet-5 的训练中，验证其是否能正常工作。


### 5 Bonus
此部分**选做**，感兴趣的同学可以尝试着完成。

#### 5.1 GPT
在自然语言处理（Natural language processing, NLP）中，早期使用的是循环神经网络（Recurrent Neural Network, **RNN**）。RNN 与 CNN 这样的前馈网络不同，RNN 中存在反馈和隐藏单元，使它可以「记住」之前读到的内容。为了解决深层网络中梯度消失或爆炸的问题，引入了长短期记忆（Long short-term memory, **LSTM**）。而为了解决传统 RNN 只能记住前面的问题，提出了双向的 LSTM。在此基础上引入的注意力机制（attention），使得网络能注意句子中重要位置的信息，例如允许在翻译中可以改变词语的顺序。

不久后，研究者发现只靠注意力机制而无需 RNN 或 CNN，就能达到较好的效果，这就是 Transformer 模型。与 RNN 不同的是，Transformer 模型能够一次性处理所有输入数据。注意力机制可以为输入序列中的任意位置提供上下文。这种架构允许更高的并行度，并以此减少训练时间。

以下为 Transformer 的结构：包含编码器和解码器，都由多个多头自注意力机制和全连接层堆叠而成，层间和层内存在归一化操作；输入由词嵌入向量加上位置信息得出。

![](index.assets/transformer.png)

Transformer 的详细结构可参考[原论文](https://arxiv.org/abs/1706.03762)。

2018 年，OpenAI 提出了生成预训练 Transformer 模型（Generative Pre-trained Transformer, **GPT**）。与先前基于监督式学习的 NLP 模型不同，GPT 在预训练生成阶段是无监督的（不需要标注），只在需要适应特定任务的**微调**（fine-tuning）时需要监督，降低了大规模 NLP 模型的门槛。GPT 的结构是 12 层仅包含解码器的 Transformer。一年后的 GPT-2 是对 GPT 的直接放大，参数量和数据集都增加了一个量级，参数量达到了 15 亿，取得了更好的效果和迁移学习能力。下一代的 GPT-3 达到了 1750 亿参数，生成的文章已经很难与人类写的区分出来。在一些领域，GPT-3 也**不再需要**专门的微调，而只需要提供例子等文本交互即可完成任务。大家可能熟悉的 GitHub Copilot 也是 GPT 的一个主要应用。GPT 系列模型的结构主要源于 Transformer 的 Encoder 部分。

本次实验要求训练一个 GPT-2/3 结构的模型，具体模型结构请参阅 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 和 [GPT-3](https://arxiv.org/abs/2005.14165) 的原论文。

#### 5.2 基准代码
为了方便起见，这里直接使用 Huffingface 提供的 GPTLMHead 模型，并采用随机输入进行训练。你也可以自行下载其他数据集进行预处理并训练，但是这不是本次实验的重点。
下面的代码里还提供了关于模型参数和计算量的计算公式，你可以参考这些公式来估计你的模型的大小和计算量。

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.config = GPT2Config(n_embd=hidden_size,
                                 n_layer=num_layers,
                                 n_head=num_attention_heads,
                                 n_positions=max_seq_len,
                                 n_ctx=max_seq_len,
                                 vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(self.config)
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]

class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_dataloader(
    vocab_size: int = 50257,
    seq_length: int = 1024,
    batch_size: int = 8,
    data_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    use_distributed_sampler: bool = False
):
    ids = torch.randint(vocab_size, (data_size, seq_length))
    dataset = TensorDataset(ids)
    if use_distributed_sampler:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )



def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


```
#### 5.3 多卡训练

单张GPU的显存和算力是有限的，随着模型大小的增长，我们需要多张GPU一起参与训练以获得更大的显存和更高的算力。多卡训练模型时常见的并行策略有**张量并行（Tensor Parallelism）**、**流水线并行（Pipeline Parallelism）**和**数据并行（Data Parallelism）**。

* 张量并行将模型层内的参数切分到不同设备进行计算，如在 Transformer 中，注意和多层感知器(MLP)的张量在向前和向后计算时按行或列分割。
  ![](index.assets/Tensor-Parallelism.png)
* 流水线并行将模型不同的层切分到不同设备进行计算，流水线中的每一设备接受上一节点的结果，并把自己的结果传递给下一设备。
  ![](index.assets/Pipeline-Parallelism.png)
* 数据并行则将全局批次大小（global batch size）按照流水线分组进行分割，每个流水线组都包含模型的一个副本，数据在组内按照局部批次规模送入模型副本，最后将各组得到的梯度进行加权平均得到总的梯度。
  ![](index.assets/Data-Parallelism.png)

在 pytorch、tensorflow 等框架中都存在分布式训练的模块，为了减轻工作量，此部分也允许使用 huggingface accelerate 等模型库，以及其他的分布式训练加速框架，例如

- DeepSpeed
- PyTorch Lightning
- ColossalAI

PyTorch 自身也有一些分布式训练的工具。

#### 5.4 模型规模

你需要按照下列表格中给定的模型结构参数实现模型。尝试使用

- 分布式训练策略（DP/ZeRO，PP，TP）
- 混合精度训练
- Gradient Accumulation
- Gradient Checkpointing
- CPU/NVMe Offload


等技术对你的模型进行加速，并将其与单卡训练进行对比（包括训练速度、显存占用、模型计算量等）。
你可以自行选择合适的 batch size 和训练数据量，并不需要关注 loss ，只需要保证不同的加速策略下训练的总数据量相同即可。



| Model size | Hidden size | Attention-heads | Layers | Sequence length | Learning rate |
| :---: | :----------: | :--------------: | :----: | :-------------: | :-----------: |
| 1.6B |     1600     |        32        |   48   |      1024      | 5e-4 |



## 6 实验任务与要求

1. 使用 `PyTorch` 实现最基本的卷积神经网络 LeNet-5，并在 MNIST 数据集上使用 GPU 进行训练，并对测试集进行测试。
2. 编写 `GELU` 算子，并在 LeNet-5 中使用该算子，验证其正确性。
3. 你需要提交：
    1. 全部代码
    2. 实验报告，其中需要包含：
        1. 简要实验过程
        2. 贴上训练过程的 **GPU 占用率截图**（使用 `nvidia-smi` 查看）
        3. Tensorboard **模型的损失曲线、LeNet-5 的准确率曲线等截图**
        4. 对于 LeNet-5，你需要写明测试集上的**识别正确率**
        5. 对于 Bonus，你需要写明训练时间、加速策略、加速效果
4. ***不允许直接使用各种深度学习开发工具已训练好的网络结构与参数***
5. ***本次实验依然会进行查重，如果你参考了网络上的代码请在报告中列出，并体现出你的理解，否则一经查出视为抄袭***
    - 关于 `GELU` 部分，你可以参考 `PyTorch` 中的实现，但是你需要在报告中说明实现在 `PyTorch` 源码的哪个位置。

## 参考资料

- `PyTorch` 框架 [https://pytorch.org/](https://pytorch.org/)
- `PyTorch Lightning` 框架 [https://www.pytorchlightning.ai/](https://www.pytorchlightning.ai/)
- MNIST 数据集 [http://yann.lecun.com/exdb/mnist/index.html](http://yann.lecun.com/exdb/mnist/index.html)
- LeNet-5 网络结构 [http://yann.lecun.com/exdb/lenet/](http://yann.lecun.com/exdb/lenet/)
- `PyTorch` 扩展 [https://pytorch.org/docs/stable/notes/extending.html](https://pytorch.org/docs/stable/notes/extending.html)
- Dive into Deep Learning [https://d2l.ai/](https://d2l.ai/)
- GELU 论文 [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)
- `torch.nn.GELU` [https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html#torch.nn.functional.gelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html#torch.nn.functional.gelu)
- GPT 介绍 [https://en.wikipedia.org/wiki/GPT-2](https://en.wikipedia.org/wiki/GPT-2)
