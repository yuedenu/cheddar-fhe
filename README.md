# Cheddar CNN 源码工程

本目录现已迁入 `/home/zijiand/cheddar` 的源码恢复与 LeNet 扩展成果；代码、依赖源码、模型权重、CIFAR-10 测试数据和参数均在本项目内。迁移后的构建和复测记录见 [docs/MIGRATION.md](docs/MIGRATION.md)。下文及原 `reports/` 中的精度、性能数字是迁移前的验收基线；本次实测单独保存在 `reports/migration/`。上游原始介绍和引用信息保存在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

本工程从源码构建 CKKS/CUDA 核心与 `ConvBN`、`DownSample`、`EvalReLU`，生成 `libcheddar_source.so`。默认构建与运行不读取历史共享库。算法及来源见 [docs/ALGORITHM_MAPPING.md](docs/ALGORITHM_MAPPING.md)，验收状态见 [reports/VALIDATION.md](reports/VALIDATION.md)。ResNet20 的源码构建、千图分类精度与单图性能验证已完成；LeNet 扩展说明见 [docs/LENET.md](docs/LENET.md)。

ResNet20 已完成：40 配置、显式 `--relu-range 16`，固定前 1000 张源码准确率 **92.5%**，原预编译 ParFuse 包 **92.0%**；单图同态推理中位数分别为 **1.099 秒 / 1.180 秒**（预热后 10 次测量，初始化及加解密另计）。默认范围仍为 10。最终已验证的构建目录为 `build-release`，精简结果在 [reports/RESNET20_RESULT.json](reports/RESNET20_RESULT.json)。

扩展其他模型时，主要参考以下文件：

- `include/extension/CNNLayout.h`：逻辑张量与槽位映射、权重展开、掩码、CPU 模拟，以及独立张量卷积参考。
- `src/extension/ExampleOps.cpp`：三个应用算子的同态执行、旋转需求、刷新与 level/scale 调度。
- `include/extension/ReLUCoefficients.h`、`ReLUChebyshev.h`：恢复的原始激活系数与等价稳定求值系数。
- `apps/Model.h`：ResNet20 拓扑、残差分支、池化、全连接及明文参考；`apps/Runtime.h`：权重、数据、参数与报告接口。

ResNet20 布局支持单张方形特征图、3×3 卷积或 1×1 投影、步幅 1/2，以及能放入一个密文的复用布局。LeNet 通过 `SpatialLayout.h` 扩展了有效卷积、平均池化和 Flatten 布局。网络拓扑仍是显式 C++，本次没有实现 ONNX 导入或自动代码生成。

## 构建

已验证的工具链为 GCC 11、CMake 3.25、CUDA 11.8；系统还需 libtommath、zlib 开发文件及 Python 3。依赖源码已固定在 `third_party/`，构建不需要联网。默认 CUDA 架构为 89，可通过 CMake 参数覆盖。

```sh
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

`ctest` 运行 CPU 布局和多项式等价性检查。GPU 基础测试与 bootstrap 测试单独执行，例如已验证的 40 配置：

```sh
build/basic_test --gtest_filter='*bootparam_40_json'
build/boot_test --gtest_filter='*Testbed32*bootparam_40_json'
```

只运行独立明文槽位与系数转换测试，无需 CUDA：

```sh
cmake -S . -B build-cpu -DCHEDDAR_CPU_ONLY=ON
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

## 运行

```sh
build/apps/resnet --report reports/single.jsonl
build/apps/resnet --mode ops --report reports/operators.jsonl
build/apps/resnet --mode relu --report reports/relu.jsonl
build/apps/resnet_acc --mode plain --report reports/plain1000.jsonl
```

`resnet` 默认一张，`resnet_acc` 默认 1000 张，两者共用模型初始化和推理实现。支持 `--params FILE`、`--data FILE`、`--weights DIR`、`--start N`、`--count N`、`--layers N`、`--report FILE`、`--relu-range B`。数据默认使用 `data/cifar10/test_batch.bin`，权重使用 `data/weights`。`--layers 10` 保存前十张完整逐层张量，所有图片均保存 logits、预测和明文参考。初始化包含明文编码与密钥生成，通常需要数分钟。

范围默认保持历史值 **10**。固定前 1000 张中的部分激活超出此范围，报告会列出图片、层和坐标并判定失败，不会静默裁剪、跳过验收或放宽容限。较大范围通过选项显式指定；它改变多项式近似对应的归一化输入，必须重新验证。

四配置单样本与独立千图长任务入口：

```sh
python3 scripts/validate.py --build build --suite smoke
python3 scripts/validate.py --build build --suite 1000
```

千图任务会保存中间 JSONL，任何超差或无效激活使退出码为 2。准确率报告中的 `count` 是请求图片数，`invalid_images` 单独报告，不能把有效样本的结果当作全量验收通过。

## 历史对照

原始头文件、库、测试及参数保存在 `legacy/snapshot`，有 SHA256 指纹。旧构建缓存、应用、模型资源和实验脚本归档在 `legacy/original-layout`。它们不在默认构建依赖中。

Git 仓库不提交构建目录和预编译库（包括 `legacy/` 下的 `.so`）；这些文件仅保留在本地。新克隆可直接按上文从源码构建。启用以下历史对照前，需自行准备与 `legacy/snapshot/SHA256.json` 指纹一致的历史库，并放到 `legacy/snapshot/lib/` 对应路径。

```sh
cmake -S . -B build-reference -DCHEDDAR_LEGACY_REFERENCE=ON
cmake --build build-reference -j 4
build-reference/apps/resnet_legacy --report reports/history.jsonl
python3 scripts/compare_reports.py reports/single.jsonl reports/history.jsonl --output reports/comparison.json
```

历史驱动使用匹配的历史头文件，并在独立进程加载 `legacy/snapshot/lib/libcheddar.so`。新驱动仅加载 `libcheddar_source.so`，每份报告的 environment 行包含实际加载路径、设备和编译器信息。历史与新实现都单独对照普通张量明文参考；密文字节不参与比较。

## 单图性能对比

性能模式与数值验证共用 `Model::EvaluateCipher`。初始化模型/编码权重/生成密钥单独计时；单图同态推理前后同步 GPU，计时内不做明文参考、逐层解密或报告写入。另记录编码加密、解密解码和三者之和；不包含磁盘读图和模型初始化。每次输入都是同一张图片的新加密，预热结果不参与统计。计时前还会进行一次完整逐层审计。测量方法与结果见 [reports/PERFORMANCE.md](reports/PERFORMANCE.md)。

```sh
build/apps/resnet --mode benchmark --count 1 --relu-range 16 --warmup 2 --repeats 10 --report reports/timing.jsonl
python3 scripts/benchmark.py --build build --legacy-build build-reference --ranges 10 16 --warmup 2 --repeats 10 --output reports/benchmark
```

对比脚本在独立进程加载新库与历史库，串行使用 GPU，并保存每次计时、正确性结果、实际加载路径、二进制 SHA256 和设备状态。旧包的严格数值检查可能失败；报告会保留退出码及失败结果，只有完整完成且预测有限、正确的计时才用于性能比较。该比较针对相同驱动调用的两个后端，不是把旧测试程序包含检查开销的总时间当作基准。

## 关键实现差异

- 卷积与投影共享经 CPU 验证的布局计划，保持残差输入，支持原地和非原地调用。
- 公开核心的 Hoist 单分解分量内核调度遗漏已修复，并有低层级回归测试。
- ReLU 保留从历史库提取的三段多项式，改用数学等价的归一化 Chebyshev 求值，改善 CKKS 数值稳定性。转换由 `scripts/derive_relu_basis.py` 使用高精度十进制完成；运行时不读取历史库。
- LeNet 另有 `lenet`、`lenet_acc` 入口；HELR、排序等其他历史应用仍归档保存。测试客户端沿用上游实验接口。

用户已确认追加 B=16 的验证（默认值仍为 10）：

```sh
python3 scripts/validate.py --build build --suite 1000 --relu-range 16 --output reports/range16-1000
```

验证脚本使用文件锁串行运行整网任务，避免多个模型与密钥同时驻留导致显存不足。

千图对照新旧后端可增加 `--historical --legacy-build build-reference`。所有图片均记录 19 次激活前的实际解密范围，前十张额外保留完整逐层张量。查看正在运行或已经完成的报告：

```sh
python3 scripts/summarize_validation.py reports/range16-1000/1000-source-40.jsonl
```

误差定位工具可将每个算子的实际输入送入独立张量参考，区分单步误差和跨层累积误差；它不会改变整网验收判定：

```sh
build/layer_error_analysis reports/single.jsonl data/weights reports/local-errors.json
```

## LeNet 扩展

原始 LeNet 测试和权重仍保存在 `legacy` 中。新入口采用原网络形状，修正原实验的通道旋转、Flatten 重复槽位和激活归一化，复用同一个源码核心库。其使用方法、原实验问题和数据约定见 [docs/LENET.md](docs/LENET.md)。

```sh
build/apps/lenet --mode full --count 1 --report reports/lenet-single.jsonl
build/apps/lenet_acc --mode full --report reports/lenet-1000.jsonl
python3 scripts/lenet_validate.py --build build --legacy-build build-reference --mode all --count 1000 --output reports/lenet-validation
```

千图和性能对照已完成：40 参数、B=10，源码和原包后端准确率均为 **9.0%**，1000 张预测一致；单图推理中位数 **205.2 / 214.1 毫秒**，源码耗时减少约 **4.15%**。源码 1230 项数值检查全部通过；原包有 89 项中间层检查超差，最终 1000 张 logits 均通过容限。两后端共用修正后的 LeNet 驱动。

现有 LeNet 权重在当前 CIFAR-10 预处理下的千图明文准确率也仅 9.0%，全部预测类别 4。原文件没有提供训练或预处理来源，因此不能据此认定模型具备有效识别能力。完整结果、数值限制及复现方式见 [reports/lenet/VALIDATION.md](reports/lenet/VALIDATION.md) 和 [reports/lenet/RESULT.json](reports/lenet/RESULT.json)。
