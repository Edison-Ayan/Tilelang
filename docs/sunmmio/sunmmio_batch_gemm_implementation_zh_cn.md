# Sunmmio Batch GEMM：语义、实现与测试

## 1. 语义及布局

在 Sunmmio 上扩展现有 `T.gemm`，不新增 `T.batch_gemm` 或 NPU-IR op。

| 输入 | 输出 | 运算 |
|---|---|---|
| A、W、C 均为二维 | `[M,N]` | 普通 GEMM |
| C 三维，A/W 各可为二维或三维 | `[B,M,N]` | 各批独立输出；二维输入被所有批共享 |
| A/W 至少一侧三维，C 二维 | `[M,N]` | 对 batch 求和后累加到同一个 C |

A/W 均为三维时，参与本次调用的 **batch region extent 必须一致**；
三维 C 的 extent 也必须一致。比较的是 region 长度，不必是整个 parent buffer
的第一维长度。batch 是第 0 轴，矩阵是最后两轴；转置不改变 batch 轴。
不支持通过 `[1,M,K]` 隐式广播到 B 批，想共享输入应使用二维 operand。
`clear_accum` 在本次调用开始时清零一次，而非逐批清零。

三维并非任意物理布局：

- TileLang 在最后两维推导硬件矩阵布局：A 为 ZZ（MX 用 MXZZ），
  C 为 ZZ；W 根据 `transpose_B` 为 ZZ/ZN（MX 用 MXZZ/MXZNN）。
  [TileLang layout 推导](../../src/op/gemm.cc)和
  [NPU-IR MMA 校验](../../3rdparty/NPU-IR/lib/Dialect/SUVM/IR/MmaOps.cpp)
  都实施这些约束。
- rank-3 batch 维必须是**单层 flat layout dimension**，其 stride 静态，
  且可换算成整存储单元和整字节。各 operand 的物理 batch stride 可以不同；
  二维共享 operand 的 stride 为 0。
  [stride 规则](../../3rdparty/NPU-IR/lib/Dialect/SUVM/IR/OpsUtil.cpp)
  以实际物理 layout 计算，不能按逻辑 shape 和 dtype 简单相乘。
- MMA TileView 必须覆盖矩阵的完整物理 tile：例如逻辑 M=16 的 ZZ
  layout 占用 32 行时，SRAM 必须**真实分配** 32 行 carrier。
  MMA 看 `[B,32,N]`，输入/输出 Copy 仍仅搬逻辑上的 16 行。
- NPU-IR 只有对 rank-3 且 B>=2 的 TileView 才启用批量 stride/count；
  B=1 可以保留三维输出类型，但不据此产生批量 dispatch。
  MX 还要求 tile view 按序覆盖 memtensor 最后各维并满足打包块约束。

## 2. 实现路径

```text
T.gemm + BufferRegion
  -> TileLang 前端检查形状、batch extent，生成高层 GEMM
  -> Sunmmio passes 处理 partial、SRAM 数据路径、布局和归约
  -> LowerTileOp 分别 lowering GEMM 与 Copy
  -> TileLang codegen 保留 MMA 的三维 region，生成 SUVM MLIR
```

### 2.1 前端：沿用 T.gemm

调用方只需写 `T.gemm(A, W, C, ...)`。[Python 入口](../../tilelang/language/gemm_op.py)
把 buffer 或切片保留为 `BufferRegion`：最后两轴用于计算 M/N/K，
第 0 轴的 region extent 用于检查 A/W/C 的 batch 是否匹配，
二维 A/W 作为共享输入。转置只影响矩阵轴。默认生成
`tl.tileop.gemm_py`（v2）；`TILELANG_USE_GEMM_V1` 可选择
`tl.tileop.gemm`（v1），两者共享这套前端检查。Sunmmio 还要求
三维 region 的 batch extent 为静态正整数、起点为静态非负整数且不越界。

### 2.2 Passes：先处理 view，再决定布局与存储

[phase.py](../../tilelang/engine/phase.py)中的关键顺序是：

| 顺序 | pass / 部件 | 对 Batch GEMM 的影响 |
|---|---|---|
| 1 | [LegalizeSunmmioBatchGemmViews](../../src/transform/legalize_sunmmio_gemm.cc) | 将静态 partial batch（如 parent `[1:3]`）复制进零基 compact buffer，再以完整 `[0:2]` region 做 GEMM；输入 copy in，输出 copy back。`clear_accum=false` 时还要先复制 C 的原值。此时 buffer 尚未定 SRAM scope，才能选合法的搬运路径并分配真实 padding。 |
| 2 | `InferSramScope`、`LegalizeSunmmioDataPath` | 分配 A/W/C 所需的 ASRAM/WSRAM/RSRAM，并安排必要的 RSRAM stage。 |
| 3 | `SunmmioLayoutInference` | 推导最后两轴的 ZZ/ZN/MX 布局、batch 维的物理排列，以及足以覆盖硬件 tile 的 padded SRAM carrier。 |
| 4 | `LegalizeSunmmioGemm` | 三维 A/W 写入二维 C 且 `clear_accum=true` 时，在 GEMM 前只清零一次 C，再用累加模式处理各批；BF16 A 的 ASRAM 双遍也在此阶段安排。 |
| 5 | `LowerTileOp` | 分别降低 GEMM 与 Copy，得到 `tl.mma_sunmmio`、`tl.dma_copy` 等 TIR；算子测试的 `lowered.tir` 停在这里。 |

### 2.3 GEMM 与 Copy 分开 lowering

[v2 GEMM](../../tilelang/tileop/gemm/gemm_sunmmio.py)在 Python 中推导布局并构造 MMA；
[v1 GEMM](../../src/op/gemm.cc)在 C++ TileOperator 中完成对应工作。
两条路径都会让 `tl.mma_sunmmio` 的 A/W/C region 保留输入的二维或三维形状。
二维 C 的归约保持 batch MMA，而不是生成 B 条二维 GEMM；
BF16 A 的 ASRAM 双遍可能产生两条 batch MMA。

[Copy lowering](../../src/op/copy.cc)独立选择搬运方式：到 ASRAM/WSRAM
的三维 Copy，只有 batch 起点对齐、源和目标的物理布局兼容时才保留
三维 DMA，否则按静态 batch 数展开成二维 plane。DRAM/RSRAM
上未对齐的 partial Copy 也展开，避免 `[1:3]` 的起点按 region extent
缩放后误寻址成 `[2:4]`；三维 layout transform 同样按 plane 处理。
**Copy 被拆成二维，不会把三维 MMA 也拆成 B 条二维 MMA。**

### 2.4 Codegen：保留 batch 维交给后端

[codegen_sunmmio.cc](../../src/target/sunmmio/codegen_sunmmio.cc)在发出
MMA 的 A/W/C region 时指定 `preserve_region_rank=true`；
[RegionCall](../../src/target/sunmmio/sunmmio_mlir_call.cc)因此保留 B=1
时的 batch 轴，并让 MMA view 覆盖已分配的完整物理 tile。
例如逻辑 M=16、物理 ZZ carrier M=32 时，MMA 看 32 行，
Copy 仍只搬 16 行逻辑数据。codegen 将 `clear_accum` 映射为
MMA 的 `acc` 条件，并把每条 TIR MMA 生成一条 `suvm.tc.mma`。
这样交给后端的是原有 MMA op 上的二维/三维 TileView，而非 TileLang
自行逐 batch 发射 MMA；后端需要这些完整的物理 view 才能计算 batch stride。

### 2.5 MX 输入

MXFP8/MXFP4 使用同一个 `T.gemm` 和 batch MMA，不另设 MX Batch GEMM op。
布局推导在 [v1](../../src/op/gemm.cc) 和
[v2](../../tilelang/tileop/gemm/gemm_sunmmio.py) 两条路径中分别实现：
A 为 MXZZ；W 在 `transpose_B=true` 时为 MXZZ，否则为 MXZNN；
C 仍为普通 ZZ。布局只作用于最后两个矩阵轴，batch 轴保留独立的物理 stride。
MX 对齐轴按 32 元素分块，块数必须为偶数；当前不自动填充未对齐的
MX operand。TileLang 为 MX MMA 保留最后两维的完整物理 tile；
K 还必须满足后端对应数据类型的打包块要求。

MX 数据包含压缩数值与 scale，不能用逻辑 `M*K*sizeof(dtype)` 推导
batch 间隔。TileLang 保留真实 MX layout 和三维 region，供后端换算
物理字节 stride；共享的二维 MX operand 无 batch stride。输入 Copy 沿用
[Copy lowering](../../src/op/copy.cc) 的规则：布局兼容且 batch 起点对齐时
可保留三维 DMA，否则逐 plane 展开；必要的 global MX→RSRAM→WSRAM
stage 是既有数据路径。partial batch 仍通过零基 compact buffer 搬运，
不能将这部分开销算成一次原生三维 DMA。上游宿主机输入需要按 MX 布局
预先打包；此路径不是在 GEMM 内把 BF16 动态量化为 MX。

## 3. 测试与 IR 产物

[test_batch_gemm.py](../../testing/python/sunmmio/ops/test_batch_gemm.py)
仅使用公开接口 `T.gemm`，运行 Sunmmio **TIR 算子级 lowering**，不调用
NPU-IR、Gem5，也不比较数值。当前共 **11 个用例**：六种 A/W/C rank 组合，
另有 B=1、静态 partial、两种输出形态下 A/W batch extent 不匹配的诊断，
以及 W Copy 展开但 MMA 保持 batched 的检查。运行命令：

```bash
pytest -q testing/python/sunmmio/ops/test_batch_gemm.py
```

其中五组测试保存 `*.input.tir` / `*.lowered.tir` 到仓库的
`build/batch-gemm-ops-ir/`（被 git 忽略，换机器先运行上述测试）：

| case 前缀 | `input.tir` 到 `lowered.tir` 的关键变化 |
|---|---|
| `batched-b2` | 三维 A/W/C；W Copy 分成两个二维 plane，MMA region 仍是 B=2。 |
| `shared-w-b2` | W 为二维、只搬一次；三维 A/C 保留 B=2。 |
| `reduction-b2` | 三维 A/W 归约进二维 C，先清零一次，再累加。 |
| `unit-b1` | B=1 的三维 A/W/C 仍保持 rank-3。 |
| `partial-b1-e2-of4` | parent `[1:3]` 被复制到零基 compact buffer，MMA 后只写回活动 C。 |

`input.tir` 中的 `T.gemm_py` 是 `T.gemm` 默认 v2 的内部形式，
保留 `T.copy`、`shared.dyn`；`lowered.tir` 停在 `LowerTileOp` 后，显示
ASRAM/WSRAM/RSRAM、`T.dma_copy` 和 `T.mma_sunmmio`。
测试形状 M=32 的 BF16 会出现**两条** MMA：原因是 ASRAM 双遍，
不是将 B=2 展成两个 GEMM。该测试只运行相关 pass 的子集；
这批文件**不是** SUVM/LLVM，也不能据此宣称数值正确性。

### MX 编译实验与数值验证范围

2026-09-20，在当前源码 `f26f6c76`、NPU-IR `5376b534` 下，本机对
[MX 数值测试脚本](../../testing/python/sunmmio/jit/test_batch_gemm_sunsim.py)
中的两组 B=2、M=N=32、K=64 用例执行了 TileLang TIR→SUVM codegen，
并使用 `npuir-opt -suvm-device-validate` 校验：

| 用例 | 输入 | SUVM MMA 数 | 校验 |
|---|---|---:|---|
| `v2-rank3-bf16-mxfp8-trans-b` | BF16 A、MXFP8 W，W 转置 | 2 | 通过 |
| `v2-rank3-mxfp8-mxfp8-trans-b` | MXFP8 A/W，W 转置 | 1 | 通过 |

两组 SUVM 的 MX operand 都是 `!suvm.tile_view<2x32x64x!suvm.mxfp8>`，
C 都是 `!suvm.tile_view<2x32x32xf32>`：batch 维在 codegen 后仍存在。
首组的两条 MMA 不表示 batch 被展开，
而是 BF16 A 的 ASRAM 双遍。另运行了上述 BF16 ops 测试及
[二维 MX codegen 测试](../../testing/python/sunmmio/codegen/test_mx_gemm.py)，
合计 `17 passed`；二维测试覆盖 MXFP8/MXFP4，但**不是**三维 MXFP4 的验证。

这些是本机编译/校验结果，未生成 ELF、未运行 gem5、没有 MX 数值误差或
cycles 结果。数值测试脚本定义了 BF16×MXFP8、MXFP8×MXFP8、
BF16×MXFP4、共享 MX 输入、batch=16、partial 和累加等用例；
它使用预打包 codes/scale，经解码得到 FP32 参考值，比较模拟输出
（MX 容差 `atol=5e-2, rtol=2e-2`）。要取得当前数值结果，须在匹配的
远端环境按仓库 `npu-remote-test` 流程运行 `--suite mx`，记录版本、
退出码、逐例 PASS、误差与 gem5 cycles。本次指定容器的 TileLang/NPU-IR
版本与本机源码不一致，且远端已有其他 gem5 进程，未同步或并发启动测试；
脚本列出的用例不能当作已通过的数值结果。

## 4. 剩余限制

1. **动态性**：B 和 partial begin/extent 都要求静态整数；当前 NPU-IR
   从 TileView 类型取常量 count，无法在同一 ELF 上改变活动 batch。
   方案是静态 Bmax carrier + runtime count/base-offset SSA、
   入口边界检查，并让 MMA 的 CSR 和 DMA 同时只处理活动 slab。
   现在可按 B 值分别 JIT，但那是不同 ELF。
2. **Copy 与 partial 成本**：布局不兼容或起点未对齐时，rank-3 Copy
   展开会增加 IR/命令数；partial compact fallback 还增加 SRAM、DMA、
   同步。长期可定义 parent carrier + base offset/parent stride 的
   SUVM window，并在 DMA lowering 分段，但不能以未验证的性能收益
   替代实测。
3. **MX 边界**：shared MXFP8 的 `32x64` plane 物理载荷为 2560 B，
   不满足目标 route 的 1024 B 连续段对齐，NPU-IR Copy 校验失败；
   `32x128` 的 5120 B 案例通过。MX cost model 未以设备数据校准，
   精确 SRAM 容量边界和更大矩阵仍需测试。global MX 经 RSRAM
   stage 再进 WSRAM 是原有数据路径，不是本次新增的 pack 算法。
4. **前端边界**：TVM let-bound 的多维 slice alias 仍可能丢外层 extent；
   直接传 BufferRegion 的已验证路径不等于所有 slice 写法都可用。
5. **集成验收**：尚未用正式 toolchain v0.2.7+ 关闭 legacy descriptor
   兼容模式复测并检查 ELF 的 descriptor/ODMA scratch section。
   dev/main 的 ODMA unit 与同步模型也需要迁移、清缓存、重新编译和完整远端 Gem5 验证。

Fork：`https://github.com/Edison-Ayan/Tilelang/tree/feature/sunmmio-batch-gemm`。
