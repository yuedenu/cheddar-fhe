# 从 cheddar 迁移至 cheddar-fhe

迁移日期：2026-10-30。目标是保留已恢复实现及其验证约定，在公开源码仓库中从空构建目录重新编译和复测。

本次结果：独立构建与两种模型千图功能一致性均通过；ResNet20 为 92.5%，LeNet 为 9%，各 1000 张预测与迁移前全部一致。性能补测存在明显波动，严格性能一致性尚未确认。完整证据与限制见 [迁移验收报告](../reports/migration/VALIDATION.md)。

## 范围与来源

- 源目录：`/home/zijiand/cheddar`，保留原样。该目录没有已提交的 Git HEAD，本次以逐文件 SHA256 为来源锚点。
- 目标目录：`/home/zijiand/cheddar-fhe`，保留原 Git 历史（上游 `8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94`）、`paper/`、`primegen32.py`、顶层 `parameters/` 和背景文档。
- 覆盖前备份：`/home/zijiand/cheddar-fhe-migration-backup-20261030`，含原 CMake、README、Hoist 头文件及实现；上游 README 另存为 `docs/UPSTREAM_README.md`。
- 迁入 `apps/`、`include/`、`src/`、`tests/`、`unittest/`、`scripts/`、`third_party/`、`data/`、`docs/`、`reference/`、历史快照及原验证报告。文件清单、来源字节数和 SHA256 见 [manifest.json](../reports/migration/manifest.json)。
- 未复制旧 `build-*`、Git 元数据、编辑器缓存、Python 缓存和历史 `legacy/original-layout/unittest/build`。旧构建目录中的完整 CIFAR 训练归档仍留在源目录；当前测试使用的完整 10000 张测试集已独立迁入 `data/cifar10/test_batch.bin`。
- 历史 `legacy/original-layout/lib/libcheddar.so` 的绝对失效链接改为项目内相对链接 `parfuse/libcheddar_parfuse.so`；历史库字节未变。

## 模型与参数

| 项目 | 本项目位置 | 本次约定 |
| --- | --- | --- |
| ResNet20 网络 | `apps/Model.h` | 显式 C++ 网络拓扑，含已融合 BN 的权重 |
| ResNet20 权重/原 checkpoint | `data/weights/` | 原字节复制，包括 `ckpt.pth` |
| LeNet 网络 | `apps/lenet/Model.h` | 已修正的三通道 32×32 变体 |
| LeNet 权重 | `data/lenet_weights/` | 10 个原始 NumPy 文件 |
| CIFAR-10 测试集 | `data/cifar10/test_batch.bin` | 固定图片 0–999，沿用原标准化 |
| 运行参数 | `unittest/parameters/` | 30/35/40/48 配置；千图使用 `resnetparam_40.json` |
| 激活参数 | `include/extension/ReLUCoefficients.h`、`ReLUChebyshev.h` | 系数与求值方法原样迁移 |
| 原模型与测试档案 | `legacy/` | 可选历史库对照；默认构建不依赖它 |

没有可迁移的 ONNX 网络文件；模型结构由 C++ 实现表达。ResNet20 千图显式 B=16，默认仍为 10。LeNet 使用 B=10；原权重明文与同态准确率均为 9%，训练来源未确认，本次仅验证迁移一致性。

## 从源码构建

```sh
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCHEDDAR_LEGACY_REFERENCE=OFF
cmake --build build-release -j 4
ctest --test-dir build-release --output-on-failure
python3 scripts/migration_validate.py --build build-release
```

最后一条命令按顺序运行 GPU 基础测试、bootstrap、原语、四配置首图、ResNet20 千图与计时、LeNet 千图与计时。完整运行需要较长时间，状态及日志保存在 `reports/migration/`。该命令使用本目录资源与新库，并将结果与已迁入的原源码后端报告比较；不需要读取源目录或历史预编译库。每次复测应使用新的 `--output` 目录，避免混淆证据。

本次构建日志为 `reports/migration/configure.log` 和 `build.log`。本次结果写入 `reports/migration/RESULT.json`；原始历史报告保持不变，其中旧绝对路径作为当时运行的来源证据保留。

可选历史对照入口也已在本目录重新编译并确认只解析到本项目的历史快照库：

```sh
cmake -S . -B build-reference -G Ninja -DCHEDDAR_LEGACY_REFERENCE=ON
cmake --build build-reference --target resnet_legacy lenet_legacy -j 4
```

这两个目标不参与本次新旧源码迁移一致性判定，历史后端原有数值差异仍见原验证报告。沙箱内无法访问 NVIDIA 设备的首次尝试保存在 `reports/migration/sandbox-attempt/`；正式 GPU 复测在可访问设备的执行环境中运行。

如单项因资源不足而中断，可用 `python3 scripts/migration_validate.py --build build-release --resume` 续跑。脚本先验证输入、共享库和已完成程序的 SHA256，保留前一轮失败证据，只复用指纹一致且成功结束的命令，重新运行失败项及结果比较。

同机连续性能对照可选用原目录的源码构建作为基线：

```sh
python3 scripts/migration_benchmark.py --baseline-build /home/zijiand/cheddar/build-release --build build-release
```

该可选命令串行测量迁移前后两个模型，各预热 2 次并测量 10 次；记录 GPU 状态与加载路径。只有这项对照显式读取原目录的程序，默认构建与模型推理不需要原目录。
