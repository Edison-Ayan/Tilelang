# Sunmmio Batch GEMM 实现、验证与剩余缺陷

本文是 TileLang 适配 NPU-IR PR #319 Batch GEMM 的唯一状态文档，合并原设计、已知问题、
Gem5实验、服务器运行和个人环境记录。文档状态日期为 2026-09-14。

## 1. 当前结论

静态 Batch GEMM 主路径已经可用：

- BF16扩展矩阵：Gem5 `22/22 PASS`。
- MX输入首组矩阵：Gem5 `4/4 PASS`。
- 静态 partial/nonzero-min batch view：Gem5 `2/2 PASS`。
- 动态 batch近期方案：按值 JIT专门化测试 `1/1 PASS`。
- 核心 codegen、MX和DMA回归：pytest `73 passed`。

仍未关闭的是同一 ELF 的 true dynamic batch、动态 partial min/extent、partial fallback和
rank-3 plane展开的性能问题、MX cost model及更大覆盖矩阵，以及正式 toolchain v0.2.7+
不使用 legacy descriptor兼容模式的复测。

## 2. 版本与环境

| 项目 | 值 |
|---|---|
| TileLang目标分支 | `dev/codegen` |
| TileLang基线 | `b7b0ca69a400deff86ff903cb71a658d5e510ab4` + 当前 Batch GEMM worktree |
| NPU-IR | PR #319 head `5376b534e670ed09c0d3dccaeab0e0d579958609` |
| 旧NPU-IR pin | `0da058e44638301b82806fe2b1cd8aa5e701a58a` |
| 服务器 | `sunmmio-m3-test`，用户 `herunkai` |
| 持久容器 | `batch-gemm-current-herunkai` |
| 容器策略 | image `tilelang-env:latest`，restart `unless-stopped`，没有 `--rm` |
| 服务器源码 | `/home/herunkai/tilelang-sunmmio/Tilelang-batch-gemm-validation` |
| 容器内源码 | `/workspace/tilelang-samples/third_party/Tilelang` |
| LLVM/MLIR | `/home/shared/llvm-install`，只读挂载 |
| 当前clang | `17.0.2 nightly-20260614-57-g811dbe4f5e` |
| Sunsim | `0.0.2` |
| Gem5 mesh | `4x4` |
| 恢复包 | `/home/herunkai/tilelang-sunmmio/bundles` |

隔离源码是从新基线同步到旧submodule worktree的，`.git`仍指向原submodule metadata，
因此服务器中直接执行 `git rev-parse` 可能显示旧的 `c9331b6d`/`0da058e`。实际版本以上表、
恢复manifest和源码checksum为准。

## 3. 接口和语义

Batch GEMM继续扩展现有 `T.gemm`，不新增 `T.batch_gemm`、Batch TileOperator或新的
NPU-IR op。

语义由A/W/C rank决定：

- A/W/C均为rank-2：普通GEMM。
- A/W任一为rank-3且C为rank-2：沿batch axis归约，输出 `[M, N]`。
- C为rank-3：输出 `[B, M, N]`。
- rank-2 A或W表示整个batch共享该operand；rank-3表示每个batch使用独立matrix slab。
- C为rank-3时，rank-3 A/W的batch extent必须与C一致。
- 不支持rank-3 extent=1的隐式broadcast；需要共享时直接使用rank-2 operand。
- batch axis固定为axis 0，最后两个轴为matrix维。

lowering契约为：

```text
T.gemm
  -> tl.tileop.gemm / tl.tileop.gemm_py
  -> tl.mma_sunmmio
  -> suvm.tc.mma
  -> A4E TensorCore LLVM intrinsics
```

NPU-IR PR #319没有新增Batch GEMM op，而是让已有 `suvm.tc.mma` 接受rank-2/rank-3
TileView组合，并扩展rank-3 DMA、batch stride、batch reduction和CSR batch count lowering。

## 4. TileLang到Gem5工具链

当前Gem5验证不需要SuDeck、Subase、torch-sunmmio或真实NPU运行时。

```text
TileLang Python kernel
        |
        v
TileLang/TVM TIR + libtilelang passes
        |
        v
kernel.mlir (SUVM)
        |
        | npuir-compile
        v
kernel.ll (LLVM IR)
        |
        | Sunmmio clang++
        v
kernel.o + main_thunk.o
        |
        | clang++ driver + lld
        v
kernel.elf
        |
        | sunsim.run()
        v
gem5.opt (4x4 A4E)
        |
        v
per-core dumps + stats.txt -> Sunsim reassemble -> NumPy comparison
```

### 4.1 TileLang frontend和passes

Python `@tilelang.jit` 先按调用参数构造静态TIR。`libtilelang.so`中的C++/TVM passes负责：

- `LegalizeSunmmioBatchGemmViews`：把静态partial view改写为compact allocation。
- `SunmmioLayoutInference`：推导ZZ、ZN、MXZZ、MXZNZ等物理layout和SRAM scope。
- `LowerTileOp`：把GEMM/Copy降为 `tl.mma_sunmmio`、DMA、transform和token操作。
- Sunmmio codegen：把device TIR翻译成SUVM `kernel.mlir`。

### 4.2 NPU-IR工具

正常JIT链路实际调用：

```bash
npuir-compile \
  --target=sunmmio-a4e \
  --emit=llvm-ir \
  kernel.mlir \
  -o kernel.ll
```

`npuir-compile`内部顺序包括canonicalize/CSE、device validation、tile split、tile
load/store和broadcast/reduce legalization、token SSA、wait/barrier解析、TensorCore init、
kernel ABI生成、SUVM-to-LLVM conversion及LLVM IR translation。

另外两个NPU-IR工具主要用于调试：

- `npuir-opt`：单独运行/打印某个MLIR pass，定位是哪一步产生错误IR。
- `npuir-translate`：把已经lower到LLVM dialect的MLIR翻译成文本LLVM IR。

正常JIT不再分别shell调用这两个工具，它们的能力已由 `npuir-compile`一站式驱动。

### 4.3 Sunmmio clang和lld

kernel LLVM IR编译命令等价于：

```bash
clang++ -c \
  --target=riscv64-sunmmio-elf \
  -mno-relax -O2 -mcpu=sunmmio-a4e \
  -o kernel.o kernel.ll
```

TileLang同时生成 `main_thunk.cpp`，其中提供 `main()`、`.kernargs`、`.descriptors`和PWLN
初始化，再用 `-x sunmmio`编译为 `main_thunk.o`。最终通过clang driver调用lld：

```bash
clang++ --target=riscv64-sunmmio-elf -mno-relax -O2 -mcpu=sunmmio-a4e \
  kernel.o main_thunk.o \
  -fuse-ld=lld -Wl,--no-warn-mismatch,--gc-sections -lnosys \
  -o kernel.elf
```

clang负责A4E指令选择和目标object生成；lld负责代码、DTCM、SRAM、kernargs、descriptor和
ODMA scratch section布局。

### 4.4 Sunsim和Gem5

`sunsim.run(elf=..., args=..., mesh=(4, 4))` 执行以下工作：

1. 从ELF的 `.sunmmio.kernel_meta` 读取kernel ABI。
2. 为Input/Output/Inout分配每核DRAM地址。
3. 按placement和ZZ/MX物理layout把NumPy数组分成16份，生成per-core `.dat`。
4. pack参数并patch ELF中的 `.kernargs`/`.descriptors`。
5. 生成 `gem5_config.yml`、InitFile和DumpFile。
6. 启动 `$SUNMMIO_GEM5/build/RISCV/gem5.opt` 和A4E config。
7. 读取16核dump，逆layout并重组输出；从 `stats.txt` 读取cycles。
8. 测试脚本使用NumPy FP32 reference执行 `assert_allclose`。

每次运行目录会保留 `invocation.sh`、YAML、输入输出dat、Gem5 stdout/stderr及m5out，可直接
重放或定位模拟器失败。

## 5. 已解决问题

| 编号 | 原问题 | 解决方式 | 验证状态 |
|---|---|---|---|
| BG-001 | `M=16`逻辑view没有覆盖ZZ layout的32行物理carrier | SRAM按layout storage size真实分配；MMA view覆盖physical extent，DMA仍访问logical extent | v1/v2 LLVM和Gem5通过 |
| BG-002 | 动态batch在深层失败且诊断不稳定 | 静态target validation，并提供按batch值JIT专门化 | 诊断和cache-key测试通过 |
| BG-003 | 高阶singleton A/W legacy GEMM被误判为Batch GEMM | 仅在合法legacy路径squeeze singleton | v1/v2回归通过 |
| BG-004 | 缺少端到端数值证据 | 建立BF16、MX、partial Sunsim/Gem5矩阵 | BF16 22/22、MX 12/12、partial 2/2 |
| MX-001 | global MX直接进入WSRAM触发1024-byte segment alignment错误 | 强制经RSRAM staging，再进入WSRAM/ASRAM | MX正向矩阵12/12通过；单plane 2560 B限制见REM-010 |
| VIEW-001 | partial/nonzero-min MMA view不覆盖完整parent allocation | pre-scope pass创建零基compact physical allocation，GEMM后copy back | `[1:3] of 4` 2/2通过 |
| DMA-001 | rank-3 TileView `index=1, extent=2`实际寻址 `[2:4]` | leading min不能整除extent时拆成rank-2 plane DMA | A copy-in和C copy-out通过Gem5 |
| GEMM-001 | let-bound GEMM参数被直接丢成整个buffer | 保留可用的BufferRegion参数 | 直接slice路径通过；TVM let-bound多维slice限制仍保留 |

### 5.1 Physical padding

`M=16`但ZZ layout物理M维为32时，padding必须有真实SRAM地址，不能只在shape上做逻辑
padding。当前allocation使用layout storage size申请32行carrier；MMA访问完整physical
view，输入/输出DMA只搬16行logical数据。

### 5.2 MX输入

已支持并验证：

- BF16 x MXFP8，transpose B，rank-3 output。
- BF16 x MXFP8，transpose B，rank-2 reduction。
- MXFP8 x MXFP8，transpose B，rank-3 output。
- BF16 x MXFP4，non-transpose B，rank-3 output。
- rank-2 shared MX A/W，以及rank-2 output的batch reduction。
- batch=16、128 cube、static partial MX和MX累加。

MX batch stride来自目标memory space的physical storage layout，不能从逻辑shape、dtype
bit-width或host packed blob大小手算。每个batch使用不同code/scale，测试可以检测错误广播。
operand dtype组合仍受A4E TensorCore约束；当前不支持MXFP8 activation x BF16 weight，测试
shared MX A时必须配合受支持的MX weight，不能把operand顺序对调后假定仍然合法。

### 5.3 Static partial/nonzero-min view

当前fallback为：

```text
parent[begin : begin + extent]
              | copy in
              v
compact[0 : extent] -> native Batch MMA -> compact C
                                               |
                                               | copy out
                                               v
                                  parent C[begin : begin + extent]
```

compact allocation按layout storage size分配physical padding。rank-2 shared operand不复制；
rank-3 C写入compact C后copy back；`clear_accum=false`时先copy in原C再累加。

## 6. 剩余缺陷和限制

| 编号 | 严重度 | 缺陷/限制 | 当前影响 | 关闭条件 |
|---|---:|---|---|---|
| REM-001 | 中 | 同一ELF不支持true dynamic batch | 每个batch值需要单独JIT/ELF，batch值多时增加编译和cache成本 | NPU-IR增加runtime batch-count SSA/CSR协议，同一ELF通过1/2/Bmax |
| REM-002 | 中 | partial begin/extent必须是静态整数 | 无法在运行时选择任意子batch | 动态逐plane DMA、边界检查和active-plane控制通过Gem5 |
| REM-003 | 中低 | compact partial fallback增加SRAM、DMA和同步 | 大shape可能容量不足或性能下降 | NPU-IR原生base offset/parent stride协议或性能证明fallback可接受 |
| REM-004 | 中低 | 不兼容layout或未对齐rank-3 Copy按batch展开 | 大batch增加IR、编译时间和DMA命令数 | native rank-3 transform/base-offset能力或分段策略 |
| REM-005 | 中低 | MX cost model未用设备数据校准 | ILP可能作出错误性能决策 | 用设备数据建立MX single/batch GEMM成本模型 |
| REM-006 | 中低 | MX容量极限覆盖仍不完整 | batch=16和128 cube已通过，但尚未找到各SRAM的精确失败边界 | 按memory scope扫描到边界前后，并验证诊断稳定 |
| REM-007 | 中 | TVM let-bound多维slice丢失外层extent | slice赋给临时变量再GEMM可能退化为整buffer | TVM保持BufferRegion alias，或前端禁止并给稳定诊断 |
| REM-008 | 阻塞正式验收 | 没有v0.2.7+无legacy复测 | 旧linker依靠测试兼容占位，不能证明正式ODMA scratch ABI | 新toolchain下全矩阵通过且ELF section无重叠 |
| REM-009 | 低，环境 | 服务器隔离副本Git metadata仍指旧submodule | `git rev-parse`和自动JSON版本可能错误 | 重建独立clean checkout并应用恢复包 |
| REM-010 | 中 | 单个shared MX plane为2560 B时不满足1024 B DMA segment对齐 | `32x64` shared MX A/W在NPU-IR lowering失败；`32x128`的5120 B载荷通过 | padded physical carrier或合法tail传输方案通过正反边界测试 |

### 6.1 True dynamic batch为什么不能只删除校验

当前NPU-IR从rank-3 `TileViewType.shape[0]` 取得静态batch数，并在每次TensorCore dispatch前
把常量写入CSR `0x808`。只删除TileLang的 `IntImm` 校验会在NPU-IR失败，或者更危险地写入
错误batch数。

建议接口是在现有 `T.gemm` 增加可选的 `batch_count`，而不是让动态shape同时承担容量和
运行时状态：

```python
T.gemm(A_max, W_max, C_max, batch_count=active_batch)
```

其中buffer type保持静态 `[B_max, ...]`，`active_batch`是kernel scalar。分层协议需要：

1. 仍按静态 `B_max` 分配A/W/C和静态batch stride。
2. TileLang的GEMM call、`tl.mma_sunmmio`和SUVM `tc.mma`逐层保留batch-count SSA operand；
   `tc.init`继续只保存静态shape/stride配置，不按runtime值重新分组。
3. 编译期验证scalar类型和 `B_max`；kernel入口生成 `1 <= active_batch <= B_max` 的
   runtime guard，越界必须trap或返回错误，不能静默截断。
4. SUVM-to-LLVM lowering在每次dispatch前把SSA值写入CSR `0x808`，替换当前从
   `TileViewType.shape[0]`生成常量的逻辑。
5. Copy先允许保守搬满 `B_max` 以打通正确性，再增加动态plane loop只搬
   `active_batch`；inactive C slab保持不变。
6. JIT cache key只包含 `B_max`，不包含 `active_batch`，同一个ELF连续验证
   `1/2/B_max`、0和越界值。

主要问题是硬件CSR只控制TensorCore dispatch，不自动约束DMA；因此只动态化MMA会正确但
浪费带宽，而同时动态化Copy又需要token、descriptor数量和循环lowering支持。rank-2 C的
batch reduction还必须保证只归约active slab，不能把未初始化的Bmax尾部计入结果。

### 6.2 Dynamic partial batch view设计

建议把容量、起点和活动长度分开表达：

```python
T.gemm(A_parent, W_parent, C_parent,
       batch_begin=begin, batch_count=extent)
```

parent仍有静态 `B_max` 和静态physical batch stride。lowering计算
`base + begin * parent_stride_bytes`，把 `extent`写入batch CSR；无需为每个runtime begin
创建新的动态shape类型。实现顺序为：

1. TileLang保留动态begin/extent PrimExpr，不再尝试把它们转换为 `IntImm` view range。
2. SUVM tile view增加动态base-offset SSA，类型仍记录Bmax carrier和静态stride。
3. 入口检查 `begin >= 0`、`extent >= 1`、`begin + extent <= B_max`，并防止整数溢出。
4. A/W/C使用相同active extent；rank-2 shared operand的offset保持0。
5. rank-3 C只更新active区间；rank-2 C只归约active extent。
6. Gem5覆盖begin=0、中间、末尾、extent=1/Bmax及全部越界组合。

难点是当前BufferRegion/TileView shape同时表示逻辑view和静态类型，动态extent不能直接塞进
现有类型。动态base GEP还必须进入address-materialization、alias/liveness分析；MX stride必须
继续由physical storage layout计算，不能用逻辑元素位宽手算。

### 6.3 去掉partial compact fallback的设计

目标是让MMA直接消费parent allocation中的窗口：

```text
parent allocation + physical layout
        | base offset + parent batch stride
        v
logical active window -> tc.mma
```

需要在SUVM中明确区分 `storage carrier` 和 `logical window`：carrier保存完整物理allocation、
layout和padding，window保存logical extent、base offset及active batch。verifier从“view必须覆盖
全部tiled extent”改成检查窗口映射后的每个访问都落在carrier内，TensorCore仍读取layout要求
的完整物理matrix tile。

这不会消除物理padding：例如逻辑 `[16, 32]` 的ZZ矩阵仍必须为物理32行申请真实SRAM；只是
partial batch不再复制到另一份compact allocation。主要风险是alias/liveness必须知道window与
parent共享存储，pipeline不能复用仍在使用的SRAM；`clear_accum=false`和非零C offset还需要
证明读写使用同一物理窗口。

### 6.4 Rank-3 Copy不按batch展开的设计

优先方案是在NPU-IR把rank-3 Copy规范化成一个“plane count + plane stride”的DMA计划，保持
单个IR op和token；若硬件descriptor只能描述二维传输，则在LLVM lowering或设备侧descriptor
循环中生成plane descriptor，而不是在TileLang IR中静态复制B份操作。动态batch时plane count
直接使用runtime extent。

需要同时定义：

1. src/dst各自的base offset、row stride和plane stride，不能假设两边layout相同。
2. 能合并为一个连续segment的条件，以及不能合并时的descriptor loop语义。
3. 一个logical Copy只产生一个完成token，wait必须覆盖全部plane。
4. descriptor池容量检查和分段提交，避免大batch耗尽DTCM descriptor。
5. MX tail策略：例如2560 B不能伪装成1024 B对齐传输；应使用有真实地址的3072 B padded
   carrier、硬件支持的masked tail，或合法的二级搬运路径。

主要问题是“减少IR展开”不等于“减少硬件descriptor”；如果硬件仍需每plane一个descriptor，
收益主要是编译时间和动态性。padded carrier会增加存储且要求host/RSRAM端都可安全访问padding，
masked tail则需要先确认A4E DMA硬件和NPU-IR ABI确实支持。

### 6.5 正式toolchain阻塞

NPU-IR从提交 `9dbcd3d` 起要求toolchain v0.2.7的ODMA scratch ABI。服务器当前旧linker
不认识 `sunmmio_dtcm_scratch`，部分descriptor较多的case会报告DTCM_DESC overflow。

当前 `--legacy-toolchain-compat` 只把测试thunk的descriptor占位从4096缩到3072 bytes，
临时让出1 KiB；它不修改kernel LLVM IR，但不是正式ABI。取得v0.2.7+后必须：

1. 安装到新的个人目录并确认实际 `SUNMMIO_TOOLCHAIN`。
2. 清空 `build/tilelang-cache`。
3. 不传legacy选项，运行BF16、MX和partial全矩阵。
4. 用 `llvm-readelf -S kernel.elf` 检查 `.descriptors` 与
   `sunmmio_dtcm_scratch` 地址范围不重叠。

## 7. 实现文件

| 文件 | 作用 |
|---|---|
| `3rdparty/NPU-IR` | pin从 `0da058e` 更新到PR #319 head `5376b534` |
| `CMakeLists.txt` | 支持预安装LLVM/MLIR，并确保HiGHS使用host编译器 |
| `src/op/copy.cc` | rank-3 DMA能力分派、physical carrier、partial plane拆分和layout兼容判断 |
| `src/op/gemm.cc` | Batch GEMM rank/batch/partial validation |
| `src/transform/legalize_sunmmio_gemm.cc` | physical carrier与partial compact allocation pass |
| `src/transform/legalize_sunmmio_datapath.cc` | global MX经RSRAM staging |
| `src/target/sunmmio/codegen_sunmmio.{cc,h}` | rank-preserving TileView、MX stride和Batch MMA codegen |
| `src/target/sunmmio/sunmmio_mlir_builder.{cc,h}` | rank-3 memtensor/tile-view MLIR构造 |
| `src/target/sunmmio/sunmmio_mlir_call.{cc,h}` | region rank/min/extent传递和legacy singleton处理 |
| `src/target/sunmmio/cost_model.cc` | Batch GEMM结构成本处理和MX保护 |
| `src/transform/sunmmio_pipeline_planning_ilp.cc` | Batch GEMM pipeline规划约束 |
| `tilelang/language/gemm_op.py` | GEMM BufferRegion参数保留 |
| `tilelang/tileop/gemm/__init__.py` | Python侧rank/batch/partial validation |
| `tilelang/engine/phase.py` | 注册partial view legalization顺序 |
| `tilelang/transform/__init__.py` | 暴露新pass |
| `testing/python/sunmmio/codegen/test_simple.py` | v1/v2 rank组合、reduction、carrier和诊断回归 |
| `testing/python/sunmmio/codegen/test_aligned_row_dma_copy.py` | rank-3 DMA和fallback回归 |
| `testing/python/sunmmio/common/compile_pipeline.py` | 测试编译pipeline接入新pass |
| `testing/python/sunmmio/jit/test_batch_gemm_sunsim.py` | BF16/MX/partial Gem5及JIT specialization测试 |

## 8. Gem5实验数据

### 8.1 BF16扩展矩阵

| 指标 | 结果 |
|---|---|
| 结果 | `22/22 PASS` |
| 覆盖 | v1/v2、rank-3 output、rank-2 reduction、shared A/W、transpose B、累加、F32/BF16 C、batch 1/2/4/8、M/N/K 16/32/64 |
| F32最大绝对误差 | `1.2770295e-5` |
| BF16最大绝对误差 | `0.00390625` |
| cycles范围 | `6328` 至 `8687` |
| 22 case墙钟合计 | `183.272 s` |

关键case：

| case | B/M/N/K | 结果 | max abs error | cycles |
|---|---|---|---:|---:|
| v2-rank3-m16 | 2/16/32/32 | PASS | 0.00000697 | 6819 |
| v1-reduction-m16 | 2/16/32/32 | PASS | 0.00001056 | 6680 |
| v2-rank3-shared-a-w | 4/16/32/32 | PASS | 0.00000751 | 6656 |
| v1-rank3-trans-b | 2/32/32/32 | PASS | 0.00000900 | 6429 |
| v2-rank3-batch8 | 8/16/32/32 | PASS | 0.00000881 | 8687 |
| v2-rank3-64-cube | 2/64/64/64 | PASS | 0.00001124 | 6627 |
| v1-reduction-bf16-output | 2/32/32/32 | PASS | 0.00390625 | 6504 |

### 8.2 MX矩阵

扩展后结果为 `12/12 PASS`，最大绝对误差 `0.00688934`，cycles范围 `6369`至
`7817`，墙钟合计 `96.607 s`：

| case | B/M/N/K | dtype | output | max abs error | cycles |
|---|---|---|---|---:|---:|
| v2-rank3-bf16-mxfp8-trans-b | 2/32/32/64 | BF16 x MXFP8 | rank-3 | 0.000288 | 6472 |
| v1-reduction-bf16-mxfp8-trans-b | 2/32/32/64 | BF16 x MXFP8 | reduction | 0.000246 | 6515 |
| v2-rank3-mxfp8-mxfp8-trans-b | 2/32/32/64 | MXFP8 x MXFP8 | rank-3 | 0.003242 | 6369 |
| v2-rank3-bf16-mxfp4-mn | 2/32/32/64 | BF16 x MXFP4 | rank-3 | 0.000137 | 6491 |
| v2-rank3-bf16-shared-mxfp8-trans-b-k128 | 4/32/32/128 | BF16 x shared MXFP8 | rank-3 | 0.000323 | 6772 |
| v2-rank3-shared-mxfp8-a-mxfp8-trans-b-k128 | 4/32/32/128 | shared MXFP8 x MXFP8 | rank-3 | 0.006432 | 6516 |
| v1-reduction-bf16-shared-mxfp8-trans-b-k128 | 4/32/32/128 | BF16 x shared MXFP8 | reduction | 0.000403 | 6699 |
| v2-rank3-bf16-mxfp8-mn | 2/32/32/64 | BF16 x MXFP8 | rank-3 | 0.000224 | 6490 |
| v2-rank3-bf16-mxfp8-batch16 | 16/32/32/64 | BF16 x MXFP8 | rank-3 | 0.000296 | 7817 |
| v2-rank3-bf16-mxfp8-128-cube | 2/128/128/128 | BF16 x MXFP8 | rank-3 | 0.000468 | 7047 |
| v2-rank3-bf16-mxfp8-partial-b1-e2-of4 | 2/32/32/64 | BF16 x MXFP8 | partial rank-3 | 0.000213 | 6791 |
| v2-rank3-mxfp8-mxfp8-accumulate | 2/32/32/64 | MXFP8 x MXFP8 | rank-3 accumulate | 0.006889 | 6445 |

首次shared MX探测还确认了一个失败边界：`32x64` rank-2 MXFP8物理载荷为2560 B，
NPU-IR在 `suvm.copy_async` lowering报 `worst-case contiguous segment (2560 bytes) must be
aligned to 1024 bytes`。改为K=128后载荷为5120 B，shared A/W和reduction均通过。因此当前
结论是shared MX语义已验证，但并非所有shape的DMA物理载荷都合法。

### 8.3 Partial矩阵

parent batch均为4，只处理 `[1:3]`：

| case | output | max abs error | cycles |
|---|---|---:|---:|
| v2-rank3-partial-b1-e2-of4 | rank-3 | 0.000009 | 6747 |
| v1-reduction-partial-b1-e2-of4 | reduction | 0.000010 | 6514 |

### 8.4 实验产物

```text
/home/herunkai/tilelang-sunmmio/tilelang-samples/build/batch-gemm-gem5-broad-20260914-final.json
/home/herunkai/tilelang-sunmmio/tilelang-samples/build/batch-gemm-gem5-mx-20260914.json
/home/herunkai/tilelang-sunmmio/tilelang-samples/build/batch-gemm-gem5-partial-20260914.json
/home/herunkai/tilelang-sunmmio/tilelang-samples/build/batch-gemm-gem5-mx-expanded-20260915.json
/home/herunkai/tilelang-sunmmio/tilelang-samples/build/batch-gemm-gem5-mx-expanded-final-20260915.json
```

## 9. 服务器复现

进入持久容器：

```bash
ssh sunmmio-m3-test 'docker start batch-gemm-current-herunkai'
ssh -t sunmmio-m3-test 'docker exec -it batch-gemm-current-herunkai bash'
```

容器内初始化环境：

```bash
source /workspace/tilelang-samples/.venv/bin/activate
source /workspace/tilelang-samples/build/env.sh
cd /workspace/tilelang-samples/third_party/Tilelang
```

修改C++后重建：

```bash
cmake --build build -j2
```

旧toolchain下运行当前全部case：

```bash
rm -rf /workspace/tilelang-samples/build/tilelang-cache
python testing/python/sunmmio/jit/test_batch_gemm_sunsim.py \
  --suite all \
  --legacy-toolchain-compat \
  --tilelang-revision b7b0ca69a400deff86ff903cb71a658d5e510ab4+batch-gemm-worktree \
  --npu-ir-revision 5376b534e670ed09c0d3dccaeab0e0d579958609 \
  --results-json /workspace/tilelang-samples/build/batch-gemm-gem5-all.json
```

可用 `--suite broad|mx|partial` 或 `--case <name>` 缩小范围。修改编译器、NPU-IR或toolchain
后必须清理明确的 `build/tilelang-cache`，否则cache hit会跳过codegen或复用旧ELF。

## 10. 持久化与恢复

- 源码、实验JSON和bundle均在宿主机 `/home/herunkai`，停止或删除容器不会删除这些mount。
- 容器使用 `unless-stopped`，服务器重启后会自动拉起，除非此前显式stop。
- 不使用 `docker commit` 保存源码；避免在97%使用率的服务器磁盘上复制完整源码/build。
- 恢复包包含binary-safe tracked diff、未跟踪测试/文档、NPU-IR Git bundle、manifest和
  SHA256SUMS。

检查环境和恢复包：

```bash
ssh sunmmio-m3-test 'docker inspect batch-gemm-current-herunkai \
  --format "status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}} image={{.Config.Image}}"'
ssh sunmmio-m3-test 'cd /home/herunkai/tilelang-sunmmio/bundles/batch-gemm-20260914 && \
  sha256sum -c SHA256SUMS'
```

恢复时应创建匹配 `b7b0ca69` 的clean checkout，恢复NPU-IR `5376b534`，应用tracked patch，
再展开untracked archive。不要直接覆盖已有脏worktree。

## 11. 后续实施顺序

1. 取得正式toolchain v0.2.7+，关闭legacy模式复跑全部BF16/MX/partial case并检查ELF section。
2. 扩展shared rank-2 MX、batch和SRAM容量边界矩阵，测量native/plane DMA性能。
3. 用设备数据校准MX和Batch GEMM cost model。
4. 修复或稳定拒绝TVM let-bound多维slice。
5. 若partial fallback性能不可接受，在NPU-IR实现native base-offset和parent-stride协议。
6. 独立设计并实现runtime batch-count SSA/CSR协议，完成同一ELF true dynamic batch。
