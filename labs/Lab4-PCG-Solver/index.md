# 实验四：PCG Solver

## 1 实验简介

[**PCG**](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method)（英语：**preconditioned conjugate gradient**，预处理共轭梯度算法）是一种利用多次迭代对方程组进行求解的方法。相比于使用直接法求解方程组，其对于**存储空间的要求不高且扩展性良好**，在涉及方程组求解的科学计算应用中具有一定的优势。大规模方程组求解在高性能计算机上进行计算时，使用直接法往往会因为程序的崩溃而导致之前的计算成为无效计算。再次进行计算时，之前已经计算过的部分要进行重新计算，浪费了大量的时间与计算资源，使用PCG算法能够有效地解决这一问题。

**PCG**涉及到的应用领域广泛，这些应用涵盖了水工、土建、桥梁、机械、电机、冶金、造船、飞机、导弹、宇航、核能、地震、物探、气象、渗流、水声、力学、物理学等几乎所有的科学研究和工程技术领域，广泛的应用于求解热传导、电磁场、流体力学等连续性问题。在商用软件中也得到了广泛的应用，如ANSYS、EPANET、MODFLOW、MATLAB、FLUENT等。

本次实验中，你需要使用OpenMP和MPI在多节点CPU集群上对串行PCG程序进行加速，并使用Profile工具进行性能分析。使用Fortran完成本次实验将可以获得Bonus。

## 2 实验环境

本实验提供了一个使用 `slurm` 管理的集群。集群的配置为四节点（GPU[01-03]，NAS00），每个节点有24个CPU核心。每次使用限制时长为15分钟。注意，虽然允许使用最多四个节点，但并不是一定要使用四个节点完成作业；如果在更少的节点上能够获得更好的效果，也可以选择使用更少的节点数，并在最终的实验报告中说明需要使用多少节点。

#### 2.1 登录

登录方式：

``` bash
ssh <username>@clusters.zju.edu.cn -p 80
```

其中 `username` 为 `{你的姓名缩写}` ，例：王小明的用户名为 `wxm`。

> 登录节点为内存 20GB 的家用 pc，请不要使用 vscode-remote-ssh 等对内存消耗大的工具

#### 2.2 编译

集群上安装有`Intel oneAPI`，启用的方式为

```bash
source /opt/intel/oneapi/setvars.sh
```

使用该命令之后会加载`Intel MPI`的相关环境。

在提供的基准代码中，我们给出了一份`Makefile`，直接使用`make`即可编译。集群上也安装有`OpenMPI`和`HPC-X`，使用`module`管理，启动方式分别为

```bash
module load openmpi/4.1.5
module load nvhpc-hpcx/23.5
```

我们欢迎大家做各种尝试，并将过程写入报告。

#### 2.3 运行

运行时可能会出现`Segmentation Fault`的问题，你可以通过

```bash
ulimit -s unlimited
```

来解决。

在这之后，你可以采用`srun`，`sbatch`或`salloc`来运行程序。一般推荐前两种。

- 使用`srun`运行程序。比如，使用4节点、8任务来运行`pcg`：

```bash
srun -N 4 -n 8 ./pcg
```

目前`IntelMPI`不能使用`srun`正常运行，不推荐使用。`OpenMPI`使用`srun`启动时会得到较多的warning，忽略即可。
如果你想要使用`HPC-X`，你应该执行

```bash
LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/comm_libs/11.8/openmpi4/openmpi-4.1.5/lib/:$LD_LIBRARY_PATH srun -N 4 -n 8 ./pcg
```

- 使用`sbatch`提交脚本进行执行。一个脚本示例如下：

```bash
#!/bin/bash
#SBATCH -o out.txt
#SBATCH -N 4
#SBATCH -n 8
#module load openmpi/4.1.5
#module load nvhpc-hpcx/23.5
source /opt/intel/oneapi/setvars.sh
mpirun ./pcg
```

会产生较多的warning，忽略即可。同理，`HPC-X`需要修改一下`LD_LIBRARY_PATH`。

- 使用`salloc`分配节点，使用`ssh`登录节点后用`mpirun`运行程序。

单次任务的最大运行时间为15分钟。**在实验截止日期前一周，最大运行时间将会减少。**

## 3 实验基础知识介绍

### 3.0 CG简述

[**CG**](https://en.wikipedia.org/wiki/Conjugate_gradient_method)（共轭梯度算法）是一种用于特定线性方程组（即矩阵为正定的方程组）数值求解的算法。
首先考虑最简单的梯度下降法。假设我们要求解线性方程组
$$
\mathbf{Ax=b}
$$
对于向量$\mathbf{x}$，其中已知$n\times n$矩阵$\mathbf{A}$是对称（即$\mathbf{A}^T=\mathbf{A}$），正定（即$\mathbf{x}^T\mathbf{Ax}>0$，对于所有非零向量 $\mathbf{x}\in \mathbf{R}^n$），且为实数，而且$\mathbf{b}$已知。

构造函数
$$
f(\mathbf{x})=\frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x}-\mathbf{b}^T\mathbf{x}+c
$$
对$\mathbf{x}$求导，得到
$$
\frac{df}{d\mathbf{x}}=\mathbf{Ax}-\mathbf{b}
$$
也就是说，该函数的极值点对应的$\mathbf{x}$就是原方程的解。于是我们把一个方程求解问题转化为优化问题。梯度下降法就是每一步求出$f$的梯度，并且沿着梯度方向下降，试图找出极值点，也就是原问题的最优解。

梯度下降算法很容易陷入局部最优解。共轭梯度算法在梯度下降算法的基础上改变了下降方向的选择策略，它选择在上一步的下降方向$p_k$和当前的梯度方向$r_{k+1}$所张成的平面上重新选择当前下降的方向$p_{k+1}$，从而达到更好的收敛效果。具体的算法推导在此略过，有兴趣的同学可以参考文末的参考资料。

**CG**算法的流程：

<div align="center">
  <img src="image/CG.png" alt="CG" style="zoom:80%;" />
</div>

输入向量$\mathbf{x_0}$可以为预估解或者$\mathbf{0}$。
### 3.1 PCG简述
**CG**算法的收敛速度由系统矩阵$\mathbf{A}$的条件数$\kappa (\mathbf{A})$决定：$\kappa (\mathbf{A})$越大，收敛越慢。
如果$\kappa (\mathbf{A})$很大，通常使用预处理将原始系统$\mathbf{Ax-b=0}$替换为$\mathbf{M^{-1}(Ax-b)=0}$，使得$\kappa (\mathbf{M^{-1}A})$小于$\kappa (\mathbf{A})$，这引出了**PCG**算法。
在大多数情况下，通过预处理确保共轭梯度算法的快速收敛是必要的，如果$\mathbf{M^{-1}}$对称、正定，且$\mathbf{M^{-1}A}$的条件数比$\mathbf{A}$更好，则可以使用预条件共轭梯度法。它采用以下形式：

<div align="center">
  <img src="image/PCG.png" alt="CG" style="zoom:80%;" />
</div>

常见的$\mathbf{M}$选取至少有两种：对角线预处理和不完备的Cholesky预处理，**本实验选择对角线预处理**，即$\mathbf{M}$矩阵为原始系数矩阵$\mathbf{A}$的对角线矩阵。在知名的开源有限元软件FEAPpv中，就是采用的这种方法进行共轭梯度法预处理。一般情况下，该方法对系数矩阵严格对角占优的情况下才比较有效。

### 3.2 访存优化

`PCG`算法中最大的计算量一般集中在矩阵与向量相乘的运算上。访存优化的原理和Lab3中的访存原理是一样的，你可以类似地使用矩阵分块、数组封装、循环展开等方式提高访存局部性。

### 3.3 并行化

#### 3.3.1 SIMD

<div align="center">
  <img src="image/225px-SIMD.svg.png" alt="CG" style="zoom:80%;" />
</div>
`SIMD`是一种数据并行技术，它通过提供支持向量运算的指令，同时对一组数据执行相同的计算，从而实现空间上的并行，进而提高程序的并行度和吞吐量。当程序中出现大量完全一致的运算需要对一批数据进行处理时，你可以考虑使用 `SIMD`对其进行并行。

`PCG`算法中矩阵与向量相乘、向量点乘的运算很适合使用`SIMD`进行加速。如果你做了Lab2.5，应该会发现手写`SIMD`指令是比较麻烦的，一般我们使用编译器来进行**自动向量化**。不同编译器对于向量化有不同的开关，请自行查阅编译器的文档进行尝试。

#### 3.3.2 指令级并行

<div align="center">
  <img src="image/pipline.png" alt="CG" style="zoom:50%;" />
</div>
现代处理器一般都会使用流水线技术来同时执行多条指令的不同阶段，从而实现指令间的并行。传统流水线因为需要解决流水线中的各种冲突，不可避免的会在流水线中带来空泡，而由于现代处理器里其实还包含指令的乱序发射，出现空泡的几率已经大大降低，所以在编程时不太需要考虑这方面的问题。

#### 3.3.3 线程级并行（OpenMP）

前面介绍的并行都是对于单个物理核心的场景。扩展到单节点、多核心的场景上，我们通常使用多线程的方式（OpenMP）进行并行。

线程的一个好处是内存共享，这意味着线程间的通信开销会小不少。因此在单机上我们往往采用多线程的并行模型。不过为了保证计算的正确性，你需要确保一些地方的操作是原子的，并维护好一些同步点。

除了数据的安全性，你还可以考虑调整调度方式和并行粒度来进行优化。比如，在考虑访存连续性的前提下，一个线程连续地执行一个`for`循环的数次迭代可能会比跳跃地执行同样次数的迭代要更高效。你可以使用`OpenMP`的`schedule`子句和`size`参数来执行调整调度方式和粒度的操作，也可以手动实现。

#### 3.3.4 进程级并行（MPI）

对于分布式的计算来说，仅仅使用线程并行已经不够了。我们需要在每个节点上开启不同的进程，来激发更多的并行能力。由于进程间的内存并不共享，我们需要进程间通信来进行数据的分享；此时，你需要均衡通信的代价和真正计算的开销，考虑好每次通信的数据大小和通信的时机，避免在通信上浪费太多的时间。为此，你可以使用异步通信，尽量在计算的同时传输后续需要的数据，提高通信的效率。

### 3.4 Profile

Profiler能够提供关于程序的运行时间统计、MPI通信开销、通信复杂均衡、访存开销等一系列可能对程序优化有用的信息。在`Intel oneAPI`中，提供了`Intel Trace Analyzer and Collector`，`VTune`，`Application Performance Snapshot`等分析工具，在加载`oneAPI`的时候就已经一起加载了。通过附加不同的参数，往往可以得到程序运行不同方面的信息。具体使用方法的文档链接放在文末参考资料中。

## 4 实验内容

### 4.1 题目描述

我们提供`PCG`代码的串行版本作为[基准代码](https://git.zju.edu.cn/zjusct/summer-course-2023/HPC101-Labs-2023/-/tree/main/docs/Lab4-PCG-Solver/code)。在这个代码框架中有一部分代码不允许修改，见代码的`README`文件（如果你认为代码框架有不足之处，或是有其他值得修改的地方，欢迎向我们提出意见）。由于迭代算法的时间稳定性较差，`main.c`将在同样一组输入上重复运行`pcg`代码10次，总时间作为在该组输入上的评测标准。

本实验包含优化和Profile两个部分。

在优化部分，你需要将该版本的`PCG`代码在给定的环境下并行化。

代码在测试时将测试4组固定的输入，其中三组公布于基准代码中，相对应的矩阵$\mathbf{A}$规模分别为2001x2001、4001x4001、6001x6001，最后一组输入不公布。`PCG`算法通常用于求解大规模的稀疏矩阵，稀疏矩阵上可以采取特殊的存储方式来节省空间和加速计算。为了代码简单，我们不考虑针对稀疏矩阵进行的优化，给出的输入数据都是稠密矩阵。

我们将检查代码的合法性（包括是否修改算法流程、是否修改了禁止修改的部分、是否违规调库等），并重新编译检查是否能够正确运行。我们将在每一组输入上重复运行可执行程序3次，取3次运行的时间平均值作为优化部分的评分依据。

此外，你还需要在代码上进行Profile。依据Profile结果，你至少需要得到

- 耗时最多的三个MPI函数；
- 程序在MPI上消耗的总时间。

Profiler还会提供更多其他的信息。如果你能够获得并解释其他Profile获得的数据和图表，或者能够根据Profile的结果对程序进行有效的优化，将能够在Profile部分获得更好的分数。

最终，你需要提交：

- 所有的源代码文件。为了节省空间，**请不要提交三个输入数据的`.bin`文件**。
- Makefile / CMakeLists.txt。我们提供的基准代码中的Makefile只作为参考，你可以更改编译器、编译选项等，并在实验报告中说明理由。
- 实验报告。Profile部分的全部内容在实验报告中体现。

## 5 Bonus

本部分**选做**，感兴趣的同学可以尝试着完成。

Fortran 是世界上第一个被正式采用并流传至今的高级编程语言，它因较为容易学习而被大量用于科学计算领域的代码编写。其编译器往往会随 C/C++ 的编译器一同安装，且其编译流程和 C/C++ 类似，故其能和 C/C++ 程序进行混合编译。

在本部分，你需要使用 Fortran 完成`PCG`算法和与`main.c`功能相同的主程序，并调用原来 C 语言版本的`judge.c`中的函数完成数据的读入和计算结果的检查。最终，Fortran 的代码和 C 的代码需要进行混合编译，并生成能够正常运行的`pcg`程序。

Bonus 部分完成即有加分（完成 Bonus 部分实验要求，且能够通过编译与测试），我们将根据完成质量提供 5-10 分的加分（与 Lab 4 权重相同）。

你也可以同时提交使用 C/C++ 的实现和使用 Fortran 的实现，对比两种实现的结果，并注明使用哪个实现参与最终 Lab 4 的排名计算 Lab 4 的分数。此时建议将 Fortran 源码和 C/C++ 源码一并上交，并通过 Makefile/CMakeLists.txt 注明两种实现分别的构建与运行方式。

## 6 注意事项

1. 严禁抄袭。查重发现的按0分处理。
2. 攻击平台、恶意浪费节点资源等干扰他人的行为，以作弊论处。
3. 不得使用其他的算法来代替`PCG`算法完成方程求解；我们考察的是并行化，而非对算法本身的改造。在代码测试时会着重检查这一点。
4. 同理，请手工实现`PCG`算法，禁止直接调库。
5. 如对题目有疑惑或认为说明文档/基准代码有缺漏，请联系助教。

## 7 参考资料

[CG 和 PCG 算法实现和分析](https://blog.genkun.me/post/cg-pcg-implementation/)

[Get Started with Intel® Trace Analyzer and Collector](https://www.intel.com/content/www/us/en/docs/trace-analyzer-collector/get-started-guide/2023-1/trace-your-mpi-application.html)

[Get Started with Application Performance Snapshot for Linux* OS](https://www.intel.com/content/www/us/en/docs/vtune-profiler/get-started-application-snapshot/2021-3/overview.html#GUID-263285D4-EA1F-471F-A984-510370E59C52)

[Get Started with Intel® VTune™ Profiler](https://www.intel.com/content/www/us/en/docs/vtune-profiler/get-started-guide/2023-1/overview.html)

[Introduction to Fortran (ourcodingclub.github.io)](https://ourcodingclub.github.io/tutorials/fortran-intro/)

[Quickstart tutorial - Fortran Programming Language (fortran-lang.org)](https://fortran-lang.org/learn/quickstart)

[计算机教育中缺失的一课 · the missing semester of your cs education (missing-semester-cn.github.io)](https://missing-semester-cn.github.io/)