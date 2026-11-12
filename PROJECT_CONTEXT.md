# Cheddar 项目背景与 ONNX CNN FHE 编译器开发目标

> 2026-10-30 迁移更新：本目录已接收 `/home/zijiand/cheddar` 中恢复的核心、CNN 算子、ResNet20/LeNet、权重、测试数据和参数。当前入口见 [README.md](README.md)，迁移清单与本次复测见 [docs/MIGRATION.md](docs/MIGRATION.md)。以下正文保留 2026-10-13 调查时的历史状态；其中“缺少实现”、原路径及 Git 状态不代表迁移后的现状。ONNX 前端与自动代码生成仍属于未来目标。

更新日期：2026-10-13  
文档性质：项目背景、源码调查记录与后续开发约定。本文中的“已确认”来自文件和源码检查；“建议”是待实现的方案，不代表已有功能。

## 1. 项目目标与当前状态

用户确定的长期目标是：**针对 CNN 网络，基于 Cheddar GPU FHE 算子库开发端到端代码生成编译器，接受 ONNX 格式模型，生成调用该算子库的可构建、可执行代码。**

预期流程为：导入模型及权重 → 分析和优化计算图 → 选择 FHE 近似与数据布局 → 规划密文计算和密钥 → 生成 C++ 代码及构建资源 → 在 GPU 上执行同态推理 → 解密验证结果。

当前已有两套互补材料：

- **历史打包快照**：用户提供的 `/cheddar`，包含头文件、预编译库、应用测试、权重、数据和既有构建产物。
- **公开源码仓库**：[scale-snu/cheddar-fhe](https://github.com/scale-snu/cheddar-fhe)，包含 C++/CUDA 实现与基础测试，但缺少本地包中的部分应用级资源。

本次完成的是调研、源码持久化和文档记录；尚未实现 ONNX 前端、编译器 IR、自动规划或代码生成器。两套目录的检查均未发现已有 `.onnx` 模型（本地扫描排除了历史构建目录）。

### 1.1 工作目录与版本锚点

| 项目 | 已确认值 |
| --- | --- |
| 历史包位置 | `/home/zijiand/cheddar` |
| 公开源码持久位置 | `/home/zijiand/cheddar-fhe` |
| 文档位置 | `/home/zijiand/cheddar-fhe/PROJECT_CONTEXT.md` |
| Git origin | `https://github.com/scale-snu/cheddar-fhe.git` |
| 调查分支 | `main` |
| 调查提交 | `8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94` |
| 提交日期及说明 | 2026-04-09，Modify citation in README |
| 本地包 Git 信息 | `/home/zijiand/cheddar/.git` 不存在，无法确定其对应的上游提交 |

“历史静态打包版本”在本文指固定的文件快照；其中 Cheddar 库是 `.so` **共享库**，不能据此称为静态链接的 `.a` 库。历史包的来源关系按用户说明记录，尚无版本证据证明它与某个公开提交一一对应。

## 2. Cheddar 的定位和能力

Cheddar 是面向 CKKS 的 C++/CUDA FHE GPU 库。公开项目强调 32 位算术、rational rescaling、NTT/INTT 和算子执行序列优化；它为编译器提供密文计算后端。项目说明及许可见[固定版本 README](https://github.com/scale-snu/cheddar-fhe/blob/8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94/README.md)和[MIT License](https://github.com/scale-snu/cheddar-fhe/blob/8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94/LICENSE)。

与 CNN 编译最相关的现有接口如下。下列“用途”是后端设计参考，不意味着库已经实现对应的通用 ONNX 算子。

| 层次 | 主要类型或接口 | 编译器中的用途 |
| --- | --- | --- |
| 数据与参数 | `Ciphertext`、`Plaintext`、`Constant`、`Parameter`、`NPInfo` | 表示密文、编码权重、模数集合、层级与 scale |
| 编码 | `Encoder::Encode`、`EncodeConstant` | 按布局、层级和 scale 编码权重、偏置、掩码 |
| 基础计算 | `Context::Add/Sub/Mult`、`HMult` | 线性组合、多项式激活 |
| 旋转 | `HRot`、`HRotAdd` | 卷积、归约、槽位搬移 |
| 层级管理 | `Rescale`、`Relinearize`、`RelinearizeRescale`、`LevelDown` | 管理乘法后的密文状态及分支对齐 |
| 密钥规划 | `EvkRequest`、`EvkMap` | 汇总旋转距离及所需最高层级 |
| 线性变换 | `StripedMatrix`、`LinearTransform`、`HoistHandler` | 对角线表示、BSGS、重复旋转的计算复用 |
| 多项式 | `EvalPoly` 及其 `Compile/Evaluate` | 构建和执行多项式；这里的 Compile 不等于 CNN 模型编译 |
| 刷新 | `BootContext`、`BootParameter`、`Boot` | 为较深计算恢复可用层级 |
| 测试客户端 | `UserInterface` | 实验中的密钥生成、加密和解密 |

本地例程使用 `uint32_t`；公开测试还包含 64 位相关配置。不能把模板类型、模数素数位宽、`log_default_scale` 和最终推理精度视为同一个概念。

**客户端边界**：上游明确将 `UserInterface.h`、`Random.h` 中的客户端功能定位为测试用途。研发阶段可以用其形成实验闭环；未来实际部署的客户端安全实现需要单独设计，不能把现有测试接口直接当作部署保证。依据见[公开 UserInterface.h](/home/zijiand/cheddar-fhe/include/UserInterface.h)及上述 README。

## 3. 历史打包快照与公开源码的差异

### 3.1 资源清单

| 内容 | 历史包 `/home/zijiand/cheddar-fhe` | 公开提交 `8df8b26` |
| --- | --- | --- |
| GPU 算子实现 | 没有顶层 `src/` | 有 `src/core/`、`src/extension/`、`src/UserInterface.cu` |
| 公共头文件 | `include/core/`、`include/extension/` 等 | 有对应核心接口，另有 `include/common/`、`NTTUtils.cuh` |
| 应用算子声明 | 有 `include/extension/ExampleOps.h` | 没有该文件及对应应用实现 |
| 库文件 | Base、SeqFuse、ParFuse 三个预编译变体 | 无这些预编译产物；CMake 从源码构建共享库 |
| 测试 | Basic、Bootstrapping、ResNet、HELR、Sorting、LeNet、CryptoNet 等文件 | `BasicTest.cpp`、`Bootstrapping.cpp`、`Testbed.h`、`CMakeLists.txt` |
| 模型权重 | `unittest/resnet20_fused/`、`unittest/lenet_fused/` | 未随该提交提供 |
| 参数 | `unittest/parameters/` 下多个工作负载的 30/35/40/48 配置 | 顶层 `parameters/` 下 bootparam_30/35/40 和 bootparam_40_64bit |
| 数据与构建缓存 | MNIST 文本数据、build 下 CIFAR-10 数据及测试可执行文件 | 未提供这些打包资源 |
| 实验脚本 | `Experiment1.py` 至 `Experiment4.py`，另有 `Experiment4-1.py` | 未提供 |

本地库路径为：

- `/home/zijiand/cheddar/lib/base/libcheddar_base.so`
- `/home/zijiand/cheddar/lib/seqfuse/libcheddar_seqfuse.so`
- `/home/zijiand/cheddar/lib/parfuse/libcheddar_parfuse.so`
- 当前 `/home/zijiand/cheddar/lib/libcheddar.so` 指向 ParFuse 版本。

`Experiment1.py` 收集基础算子耗时；`Experiment2.py` 测应用耗时；`Experiment3.py` 切换三个库变体作比较；`Experiment4.py` 和 `Experiment4-1.py` 收集精度相关输出。它们会修改库软链接，因此后续复现实验需要记录实际加载的库。

### 3.2 最重要的兼容性结论

本地 `ExampleOps.h` 声明了 `cheddar::example_ops::ConvBN`、`DownSample`、`EvalReLU` 等应用算子。对 ParFuse 库的 `nm -D -C --defined-only` 检查确认，库内存在上述类的 `Evaluate` 导出符号；但当前两个目录均未找到这些算子的实现源码。

因此，**公开源码重新构建出的 Cheddar 不能直接视为历史应用库的完整替代品**。复制一个头文件不能补齐缺失实现。建议将本地 ResNet20 作为现有库的行为参考，将公开实现作为可维护后端的基础；通用 CNN 算子应逐步建立在公开的基础原语和扩展接口上。

共同头文件的字节差异也不应夸大为全部 API 不兼容。抽查 `Context.h`、`Container.h`、`UserInterface.h`、`BootContext.h`、`LinearTransform.h` 后，主要差异是 `API_EXPORT`、注释和排版。`Parameter.h` 中还有融合选项名称变化，例如本地 `kFuseAccumHoist/kFuseKeyMultHoist` 对应公开版本中的 `kFuseGSPAccum/kFuseBSKeyMult`。相近的声明不能证明 ABI 或实际行为完全相同，迁移仍需编译和数值验证。

依据：[本地应用算子头文件](/home/zijiand/cheddar/include/extension/ExampleOps.h)、[公开构建配置](/home/zijiand/cheddar-fhe/CMakeLists.txt)、[公开测试配置](/home/zijiand/cheddar-fhe/unittest/CMakeLists.txt)。

## 4. 现有 CNN 示例能提供什么

### 4.1 ResNet20：完整手写调度的主要参考

[ResNet.cpp](/home/zijiand/cheddar/unittest/ResNet.cpp) 手工组织了 CIFAR-10 输入、首层卷积、三个 stage 的九个残差块、平均池化和全连接层，最终解密获得 10 类输出。主干通道按 16 → 32 → 64 增长，空间尺寸按 32 → 16 → 8 缩小。

值得保留的实现经验：

1. **权重准备**：通过 `cnpy::npy_load` 读取 `resnet20_fused/` 中的 `*_reparam.weight/bias`。抽查 `conv1_reparam.weight` 的文件头确认其为 NumPy 格式，float32，形状为 `(16, 3, 3, 3)`；扩展名不是 `.npy` 也不代表其为裸二进制。
2. **输入契约**：CIFAR-10 数据先除以 255，再按均值 `[0.4914, 0.4822, 0.4465]`、标准差 `[0.2023, 0.1994, 0.2010]` 标准化。三通道数据补零为四通道大小后编码。
3. **槽位布局**：显式维护 width、channel、pack、repeat 等信息，跨 stage 的 pack 为 1 → 2 → 4。这些是具体算法的布局约定，不能直接等同于 ONNX 张量形状。
4. **密钥准备**：卷积、下采样、池化、全连接分别注册旋转请求，统一调用 `PrepareRotationKey`；bootstrap 也有自己的准备和密钥请求。
5. **深度管理**：残差相加前有层级调整，池化和全连接前通过 `AdjustLevelWithBoot` 判断是否需要刷新。
6. **算子组合**：池化使用 `Trace` 和 `HoistHandler`，全连接使用 `StripedMatrix` 和 `LinearTransform`。
7. **激活近似**：使用 `EvalReLU`，存在 `relu_range=10` 和分阶段 level 设置；近似系数和内部刷新行为不能只靠头文件完整恢复。

`ResNet.cpp` 默认只处理 1 张图；[ResNet_acc.cpp](/home/zijiand/cheddar/unittest/ResNet_acc.cpp) 默认 1000 张，可用 `--num_test_images` 调整，当前只实例化 `resnetparam_40.json`。因此不能将单张预测或该程序的一组输出解释为四套配置上的完整精度评估。实验脚本中固定的四列配置也需与可执行程序实际输出核对。

本地 ResNet 参数实测文件值如下，四组均为 `log_degree=16`，即多项式次数 65536、最大复数槽数 32768：

| 参数文件 | log_default_scale | default_encryption_level | level_config 项数 |
| --- | ---: | ---: | ---: |
| resnetparam_30.json | 30 | 19 | 31 |
| resnetparam_35.json | 35 | 19 | 31 |
| resnetparam_40.json | 40 | 12 | 25 |
| resnetparam_48.json | 48 | 11 | 24 |

完整模数链层数、默认加密层级、bootstrap 后层级是不同概念。可用刷新输出层级应从 `BootParameter::GetEndLevel()` 获得，不能统一写死为 8 或 12。

### 4.2 LeNet 与 CryptoNet：底层实现参考，尚不是正确性基准

| 文件 | 观察结果与价值 |
| --- | --- |
| [lenet.cpp](/home/zijiand/cheddar/unittest/lenet.cpp) | `lenet_acc` 目标；用基础原语手写 `ManualConv`、`AvgPool2x2`、`Flatten`、`GemmFC` 和激活多项式，适合研究算子展开 |
| [LeNet_acc.cpp](/home/zijiand/cheddar/unittest/LeNet_acc.cpp) | `lenet_acc_1` 目标；另一个 LeNet 实验版本，仍使用 `EvalReLU`，不能假定已经摆脱历史二进制应用算子 |
| [cryptonet.cpp](/home/zijiand/cheddar/unittest/cryptonet.cpp) | 忽视

尤其要注意：`lenet.cpp` 的输入是三通道 32×32 的常量 0.5，并非完整数据集精度测试。它采用按通道连续、空间补齐到二次幂 tile 的布局，并显式执行 Flatten 搬移。文件中还留有实验代码与相互矛盾的 scale 注释，应经算子级测试后再提取通用实现。

本次没有重新编译或运行上述 GPU 测试，不将历史 build 目录和 CSV 的存在视为正确性、性能或精度已被复现。

## 5. 后端规划必须尊重的实际语义

以下依据公开实现 [Context.cpp](/home/zijiand/cheddar-fhe/src/core/Context.cpp)、[Parameter.cu](/home/zijiand/cheddar-fhe/src/core/Parameter.cu)、[EvkRequest.cpp](/home/zijiand/cheddar-fhe/src/core/EvkRequest.cpp)。

- **Mult 与 HMult 不同**：`Mult` 本身不自动完成重线性化和 rescale；`HMult` 组合乘法与重线性化，并由参数控制 rescale。编译器须知道调用后密文分量数、level 和 scale 的变化。
- **scale 随层级变化**：`GetScale(level)` 使用参数生成的逐层 scale；`Rescale` 除以 `GetRescalePrimeProd(level)`。不能在整张图里统一假定每层 scale 都等于 `2^40`，也不能假定每层只对应删掉一个素数。
- **LevelDown 不是简单删除素数**：公开实现逐层乘编码常量 1，再执行 `Rescale`。其行为与 `lenet.cpp` 中“LevelDown only removes primes, not scale”的注释不同。编译器应以目标版本实现及测试为准；历史二进制中的行为仍需单独确认。
- **改变元数据不等于数值转换**：`SetScale`、`SetNumSlots` 不应被当成任意重新缩放或重新排布数据的通用实现。
- **旋转请求含层级**：`EvkRequest::AddRequest(rot, level)` 对同一旋转距离保留最高层级；仅收集距离集合不足以完整生成密钥计划。
- **bootstrap 有前提**：须准备 EvalMod、相应槽数的 FFT 和密钥，再调用 `Boot`。规划要同时考虑允许的槽数、布局、数值范围、刷新输出层级及成本。

这些约束意味着编译器需要显式维护密文状态，不能只将一个 ONNX 节点替换为一个库函数名。

## 6. 建议的编译器范围与产物

本节是开发建议；用户已确定 ONNX CNN → Cheddar 代码生成这一方向，具体实现语言、IR 框架、opset 和性能目标尚未确定。

### 6.1 首期边界

建议首期采用静态形状、batch=1、NCHW、明文模型权重、加密输入的推理场景；先将 Conv 限定为明确支持的二维卷积属性组合，逐步覆盖 stride、padding、dilation 和 group。模型权重保密、动态控制流、大 batch、跨密文大张量等列为后续扩展。

ONNX 导入必须读取 opset、属性、initializer、输入输出类型与形状，执行模型检查和形状推导。推导不出形状或遇到不支持的语义时，应明确诊断，不能猜测。[ONNX 概念文档](https://onnx.ai/onnx/intro/concepts.html)与[形状推导文档](https://onnx.ai/onnx/repo-docs/ShapeInference.html)可作为前端依据。

### 6.2 首批算子与后端方向

| ONNX 算子/模式 | 建议处理方式 | 关键限制 |
| --- | --- | --- |
| Conv | 旋转、明文权重乘法、累加、偏置；后续评估 hoisting | 显式处理 kernel、pads、strides、dilations、group 及跨通道槽位映射 |
| Conv + BatchNormalization | 推理模式且参数为常量时折叠权重和偏置 | 保留 epsilon、图分支及模型数值语义 |
| Add | 密密或密明加法 | 检查广播、布局、level 与 scale |
| Relu | 显式选择多项式近似，由 EvalPoly 或基础原语实现 | 标注拟合区间、误差和深度；不是精确等价转换 |
| AveragePool / GlobalAveragePool | 旋转归约、掩码、缩放和必要的重排 | 处理 padding、除数和输出位置 |
| Flatten / Reshape | 能保持映射时只改逻辑视图，否则生成搬移 | 张量形状变化不能自动假定为零成本 |
| Gemm / MatMul | 常量权重线性变换，必要时加偏置 | 尊重转置、alpha/beta、广播和维度 |
| Identity / 常量计算 | 删除冗余节点或常量折叠 | 保留输出契约和图分支语义 |

MaxPool、Softmax、ArgMax 等应单独定义策略。不能无提示地将 MaxPool 改成 AveragePool，或删除 ONNX 声明的 Softmax 输出。可以先以输出 logits 的模型作为支持范围，解密后做分类决策，但这必须是明确的输入输出契约。

### 6.3 建议流水线

```mermaid
flowchart LR
    A["ONNX 模型与权重"] --> B["校验 / 形状推导 / 图规范化"]
    B --> C["CNN IR 与激活近似"]
    C --> D["槽位布局 / 密文切分"]
    D --> E["FHE IR：level、scale、密钥、bootstrap"]
    E --> F["Cheddar C++ 与构建资源"]
    F --> G["GPU 密文推理"]
    G --> H["解密结果与明文基准对比"]
```

建议至少区分两级表示：

- **CNN IR**：节点拓扑、形状、dtype、布局、常量参数、算子属性、激活近似及其取值范围。
- **FHE IR**：密文/明文值、槽位映射、密文数量、level、scale、模数参数标识、密文分量数、旋转、重线性化、rescale、bootstrap、密钥需求与存活区间。

第一版可用小型自定义 IR 验证流程；是否采用 MLIR 留待规模和优化需求明确后决定。前端与代码生成语言尚未锁定。

期望一次编译生成：

1. C++ 模型执行函数与初始化代码，调用固定版本的 Cheddar。
2. 权重、偏置、掩码和布局资源，以及编码所需的 level/scale 信息。
3. 参数配置、旋转和其他评估密钥清单、bootstrap 计划。
4. CMake 工程或等价构建入口。
5. 测试驱动及编译报告，记录不支持算子、近似策略、密文数量、运算统计和版本。

运行时可以加载编译器已生成的权重资源并编码；首期不必强制离线持久化所有 GPU 明文对象。“端到端”的验收重点是从 ONNX 到可构建执行产物的自动化，不能仍要求手工重写每层推理流程。

## 7. 建议的开发顺序与验收方式

| 阶段 | 交付 | 验收依据 |
| --- | --- | --- |
| M0：固定可复现后端 | 明确使用的库、头文件、参数、构建环境和样例资产 | 原语及 bootstrap 的数值测试；记录库与参数版本 |
| M1：最小代码生成闭环 | 小型 ONNX CNN → 图表示 → C++ → 构建执行 | 不手改生成代码即可运行，并匹配对应明文模型 |
| M2：通用 CNN 算子 | Conv、池化、Gemm、激活近似、形状变换及残差 Add | 随机输入及边界形状的逐算子/逐层对比 |
| M3：自动 FHE 规划 | 布局、密钥、level/scale、bootstrap 与临时密文管理 | 计划合法、无缺失密钥；数值误差和资源消耗可解释 |
| M4：ResNet20 回归 | 用对应 ONNX、权重与预处理自动生成整网 | 对比原始明文模型、近似模型、历史手写实现及新后端 |
| M5：性能优化 | hoisting、BSGS、编码复用、内存复用、批处理等 | 在相同模型、参数、精度与设备下比较性能 |

验证至少区分三类结果：

- 原始 ONNX 模型与采用多项式等变换后的明文模型：衡量模型近似造成的误差和精度变化。
- 变换后的明文模型与 FHE 解密输出：衡量编译、布局与同态数值误差。
- 数据集层面的分类精度、单样本延迟、吞吐、峰值显存、密钥体积和 bootstrap 次数：衡量整网可用性及成本。

误差容限、准确率下降上限和性能目标尚未约定。建议先报告逐层误差，再讨论优化，避免只靠最终 argmax 掩盖中间计算错误。性能统计应区分模型/密钥准备、加密、GPU 推理和解密，明确预热次数与样本数量。

## 8. 构建背景与待解决问题

公开构建使用 C++17/CUDA17、CMake ≥ 3.24、CUDA Toolkit ≥ 11.8；README 建议 Pascal 或更新 GPU，至少 16 GB 显存。依赖包括 RMM，默认使用 libtommath，也可启用 GMP；测试 CMake 实际使用 GoogleTest 与 nlohmann/json，不能被其中名为 `jsoncpp` 的 FetchContent 变量误导。

本地测试额外使用 cnpy 和 Eigen，并写死了 `/home/zijiand/cheddar/include`、`/home/zijiand/cheddar/lib`、`/home/zijiand/cheddar/unittest/build/CPM.cmake` 等路径。现有 build 目录是历史环境产物，后续需要把这些路径和依赖改造成独立、可复现的构建输入。

后续工作的已知缺口：

- 历史包精确版本、编译器/CUDA/ABI 及三个库变体与公开优化选项之间的对应关系尚未确认。
- `ConvBN/DownSample/EvalReLU` 有声明与二进制符号，缺少可维护实现源码。
- 已有权重缺少一套已核验的“模型定义—导出脚本—ONNX—预处理—明文结果”对应链。
- LeNet 手写原语实现、激活系数与 scale 处理需要验证，不能直接升级成默认 lowering。
- 需要核验历史参数在公开版本上的行为，以及参数选择的精度、资源与安全约束。
- 本次没有建立新的构建环境，也没有确认现有二进制在当前设备上能够正确执行。
- 权重和数据的来源及使用条件不能仅由公开源码的 MIT 许可推断。

## 9. 后续接手时的阅读顺序

1. 本文，确认目录、版本、已知缺口及长期目标。
2. [公开 README](/home/zijiand/cheddar-fhe/README.md)和[根构建文件](/home/zijiand/cheddar-fhe/CMakeLists.txt)，理解后端边界。
3. [本地 ResNet.cpp](/home/zijiand/cheddar/unittest/ResNet.cpp)与[ExampleOps.h](/home/zijiand/cheddar/include/extension/ExampleOps.h)，理解历史完整应用。
4. [本地 lenet.cpp](/home/zijiand/cheddar/unittest/lenet.cpp)，研究显式布局和原语展开，同时检查其数值假设。
5. [公开 Context 实现](/home/zijiand/cheddar-fhe/src/core/Context.cpp)、[Parameter 实现](/home/zijiand/cheddar-fhe/src/core/Parameter.cu)和[BootContext 实现](/home/zijiand/cheddar-fhe/src/extension/BootContext.cpp)，确定可依赖的运行语义。
6. [LinearTransform](/home/zijiand/cheddar-fhe/src/extension/LinearTransform.cpp)、[Hoist](/home/zijiand/cheddar-fhe/src/extension/Hoist.cu)、[EvalPoly](/home/zijiand/cheddar-fhe/src/extension/EvalPoly.cpp)，研究可复用优化。

后续开发应保留历史包作为参照，并持续记录实际使用的公开提交、库文件、参数和模型资源。本项目的核心新增工作是将现有手写 CNN 中的图处理、布局、密钥和数值调度转化为自动编译过程。

## 附录：本次调查使用的历史库指纹

以下 SHA-256 用于区分历史二进制，不代表它们与公开源码构建结果一致：

```text
base:
1a62a9138fbb8bf9cd81c06dcf77c8a12a64bb4ed018e30d98bfce9dbc33ee4b
seqfuse:
006c31a6ea524dcbf417b6dbd299bf33654f4ace2feb318b8d2cc5063a8f907b
parfuse:
5714ed333ffa7cc8a7aec9b0ecfa0520a48376b24942afad5dd7449ad67a7aa4
```
