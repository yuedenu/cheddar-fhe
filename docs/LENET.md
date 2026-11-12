# LeNet 源码扩展与对照

## 原始资产

原有信息保留在 `legacy/snapshot/unittest/lenet.cpp`、`LeNet_acc.cpp` 和 `legacy/original-layout/unittest/lenet_fused/`，没有删除。10 个权重文件已按原字节复制到 `data/lenet_weights/`，形状和 SHA256 见 `reports/lenet/assets.json`。

这是三通道 32×32 输入的 LeNet 变体：5×5 有效卷积 3→6、ReLU、2×2 平均池化、5×5 有效卷积 6→16、ReLU、2×2 平均池化、Flatten、400→120→84→10 全连接，其中前两层全连接后有 ReLU。归档中名为 `LeNet_acc.cpp` 的文件只推理常量 0.5 图片，没有真实数据集千图循环。

本次采用现有 CIFAR-10 测试集的固定前 1000 张，与 ResNet20 相同的预处理：像素除以 255，均值 `[0.4914,0.4822,0.4465]`，标准差 `[0.2023,0.1994,0.2010]`。这是根据输入尺寸和项目现有数据选择的验证约定；原 LeNet 实验没有提供训练或预处理来源，不能宣称复现了原训练精度。MNIST 文本文件也仍在归档中，但不对应这套三通道输入接口。

## 已确认的原实验问题

- `ManualConv` 的旋转只包含输入通道和卷积核偏移，漏掉输出通道位置。首图独立张量对照的第一层最大误差为 1.727069；只有第 0 个输出通道一致。证据见 `reports/lenet/archived-formula-audit.json`。
- Flatten 仅设置 `num_slots=512`，没有形成重复的 512 槽向量。新的实现先生成实际重复副本再更新元数据。
- 使用复合 ReLU 的原实验没有将数值归一化到其近似区间。扩展采用与 ResNet20 一致的契约：首层权重和隐藏层偏置除以 B，所有隐藏张量表示真实值/B，最后全连接权重乘 B 恢复 logits。默认 B=10，当前千图明文预检无越界。
- 另一份 `lenet.cpp` 使用不同的实验三次激活。本次选择 `LeNet_acc.cpp` 的复合 ReLU 网络作为扩展依据，复用已恢复的激活算子；两份原文件保持原样。

## 源码结构

- `include/extension/SpatialLayout.h`：通道连续的补零 tile、5×5 有效卷积/池化/Flatten 稀疏对角线计划、BSGS 掩码预旋转、槽位模拟与独立张量参考。
- `apps/lenet/Model.h`：将计划编译为公开 `HoistHandler` 原语，执行真实槽位重复，接入 `LinearTransform` 和恢复后的 `EvalReLU`。
- `apps/lenet/main.cpp`：CIFAR-10 千图验证和单图计时，共用同一同态图。
- `tests/spatial_layout_test.cpp`：5 种实际形状 × 零/常量/边界脉冲/固定随机输入，共 20 个场景，检查全部补零和重复槽位，`atol=rtol=1e-9`。
- `scripts/lenet_precheck.py`：独立 NumPy 明文参考；千图 C++/NumPy logits 最大差异为约 5.1e-15。
- `scripts/lenet_validate.py`：串行运行两个后端，保留输入指纹、实际加载路径、数值检查、预测及性能样本。

核心 `libcheddar_source.so` 和 ResNet20 算子实现无需修改。新增的 LeNet 使用同一套从源码恢复的原语库。

| 阶段 | 逻辑形状 | 槽位周期 | 输入→输出 level |
| --- | --- | ---: | --- |
| 输入 | 3×32×32 | 4096 | 2 |
| Conv1 | 6×28×28 | 8192 | 2→1 |
| ReLU1 | 6×28×28 | 8192 | 刷新后→2 |
| Pool1 | 6×14×14 | 2048 | 2→1 |
| Conv2 | 16×10×10 | 4096 | 1→0 |
| ReLU2 | 16×10×10 | 4096 | 刷新后→2 |
| Pool2 | 16×5×5 | 1024 | 2→1 |
| Flatten | 400 | 512 | 1→0 |
| FC1 / ReLU3 | 120 | 512 | 先刷新至可用层，1→0；激活后→2 |
| FC2 / ReLU4 | 84 | 512 | 降至 1→0；激活后→2 |
| FC3 | 10 | 512 | 降至 1→0 |

同一修正后的驱动分别链接新源码库与历史 ParFuse 库，进程隔离。两者准备相同的 512/4096/8192 槽 bootstrap。这是原包后端对照，不把修正后的驱动声称为原始常量测试程序。

## 构建与运行

```sh
cmake -S . -B build -G Ninja
cmake --build build -j 4
ctest --test-dir build --output-on-failure
build/apps/lenet --mode full --count 1 --report reports/lenet-single.jsonl
build/apps/lenet_acc --mode full --report reports/lenet-1000.jsonl
build/apps/lenet --mode benchmark --count 1 --warmup 2 --repeats 10 --report reports/lenet-benchmark.jsonl
```

`--params FILE`、`--weights DIR`、`--data FILE`、`--relu-range B` 可显式覆盖资源，默认使用 40 参数、原 LeNet 权重和 B=10。仅做明文预检可使用 `--mode plain`，不初始化 CUDA。

历史对照单独开启；新后端默认构建不依赖历史包：

```sh
cmake -S . -B build-reference -DCHEDDAR_LEGACY_REFERENCE=ON
cmake --build build-reference --target lenet_legacy -j 4
python3 scripts/lenet_validate.py --build build --legacy-build build-reference --mode all --count 1000 --output reports/lenet-validation
```

`all` 模式在同一次模型初始化后先完成千图，再预热 2 次并测量第 0 张图片 10 次。计时区间前后同步 GPU，排除明文参考、逐层解密及报告写入；编码加密、解密解码、初始化单独记录。原包内部日志保留。由于现有模型首图预测错误，性能有效性要求输出有限并与同模型明文预测一致，不把错误标签改写成正确。

## 分类能力限制

现有权重在当前 CIFAR-10 约定下，精确 ReLU 与多项式 ReLU 的前 1000 张准确率均为 **9.0%**，全部预测类别 4。模型权重呈现接近常见随机初始化的范围，但其训练来源未确认，因此不将“未训练”写为已证实事实。此结果限制了分类精度对照的解释：即使新旧 FHE 预测一致，也不能据此证明模型具备有效识别能力；逐层数值对照仍可检验算子实现。

GPU 千图和性能验证已完成：源码和历史后端准确率均为 9.0%，预测一致率 100%；单图推理中位数分别为 205.2 和 214.1 毫秒。源码 1230 项数值检查全部通过，历史后端保留 89 项中间层超差。完整结果见 [验证报告](../reports/lenet/VALIDATION.md)。独立 NumPy 预检需 Python 的 NumPy 包，可运行 `OPENBLAS_NUM_THREADS=1 python3 scripts/lenet_precheck.py`。

对照脚本的成功退出表示源码验收通过、两后端任务完整且计时有效；历史后端的严格数值失败单独保存在 `results.json` 和 `comparison.json`，不会被改写成通过。计时模式还要求每次输出有限且预测与明文一致。
