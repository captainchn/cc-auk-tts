# cc-auk-tts

用 **AuK** 在你自己的 Mac 上生成配音，拿一段自己的录音做音色参考，给一个脚本就出一版音轨。

这套脚本解决的是**粗剪阶段的配音卡点**：视频还没定剪，但需要一版和脚本对得上的配音来定每句话占多长时间。以前这一步要么等进棚，要么用系统自带的合成语音凑合（听着不像自己，语调也不对）。现在脚本一改，几分钟就出一版。

生成的声音能听，音色接近本人，**它的用途是站位，不是成片**。成片阶段仍然建议自己重配。

## AuK 是什么

腾讯混元 2026-09-09 开源，MIT，论文 [arXiv 2609.08936](https://arxiv.org/abs/2609.08936)。

- 1.5B 的语音生成与编辑基础模型，两个权重：`AuK`（base，质量优先）和 `AuK-Flash`（4 步蒸馏，快）
- 16 类任务走同一个自然语言指令接口：零样本 TTS、指令 TTS、内容编辑、歌词编辑、音高 / 语速 / 音量编辑、情绪 / 音色 / 口音 / 非语言音 / 耳语编辑、降噪去混响、人声分离、音乐分离、目标说话人提取
- 官方仓库：<https://github.com/Tencent-Hunyuan/AuK>
- Apple Silicon 走官方 MLX 后端：`docs/MLX.md`（**不是** PyTorch MPS）

## 两个必须知道的硬约束

**一、单次生成 ≤ 30 秒。** 这个 30 秒是「参考音频 + 生成目标」加起来的预算。一段 178 字的稿子读出来约 32 秒，所以长稿必须切段。本仓库的 `run_auk.py` 就是干这件事：按轨单逐段生成，再拼成整条。

**二、输出时长是硬控的。** 给 `gen_seconds` 多少就出多少，不随机。这条既是这套东西能用来「站位」的原因，也意味着它同时定了语速。经验值约 **0.19 秒/字**；给宽了语速就慢，听感会拖。

## 安装

按官方 `docs/MLX.md` 走，这里只记三个我在 macOS 上实际踩到、官方文档没写的坑。

```bash
git clone https://github.com/Tencent-Hunyuan/AuK
cd AuK
git checkout feat/mlx-apple-silicon

uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e . && uv pip install mlx
uv pip install "transformers>=4.52,<5"

# 下载权重（base + Flash + 编码器）
hf download tencent/AuK            --local-dir ./ckpts/AuK
hf download tencent/AuK-Flash      --local-dir ./ckpts/AuK-Flash
hf download Qwen/Qwen2.5-Omni-3B   --local-dir ./ckpts/Qwen2.5-Omni-3B

# 转一次权重（torch → MLX 布局），约写 28 GB fp32
export PYTHONPATH=src
python -m auk_mlx.convert vae     ckpts/AuK/vae.safetensors            ckpts/mlx/vae.safetensors
python -m auk_mlx.convert dit     ckpts/AuK/auk_base.safetensors       ckpts/mlx/dit_base.safetensors
python -m auk_mlx.convert dit     ckpts/AuK-Flash/auk_flash.safetensors ckpts/mlx/dit_flash.safetensors
python -m auk_mlx.convert thinker ckpts/Qwen2.5-Omni-3B                ckpts/mlx/thinker
```

### 三个坑（都花了真时间）

**1. `uv venv --python 3.10` 可能卡死在下载 CPython 上。**
表现是十几分钟没有任何输出。不是死机，是在下编译器工具链。本机没有 3.10 时，可以直接用系统已有的 3.11，`requires-python >=3.10` 满足。

**2. `antlr4-python3-runtime 4.9.3` 只有源码包，没有预编译轮子。**
它是 `omegaconf` 的依赖。本地构建过程中要批量删 52 个文件，会撞上环境的删除保护（默认阈值 50）导致失败。解法是手工把这个纯 Python 包打成 wheel 再装：

```bash
pip download --no-binary :all: antlr4-python3-runtime==4.9.3 -d /tmp/antlr_wheel
# 解包，把 src/ 布局压成 wheel，再 uv pip install <wheel>
```

**3. `uv pip install -e .` 装 auk 包本身可能报 `EEXIST`。**
构建目录有残留时会发生。**这一步其实可以跳过**：跑推理并不需要把包装进环境，`PYTHONPATH=src python -m auk_mlx.cli` 就够了。只装依赖清单即可。

## 跑起来

```bash
export AUK_REPO=/path/to/AuK

# 1. 从自己的录音里裁一段参考（8 到 20 秒，削波 0、静音少）
python3 prep_ref.py --src 我的录音.wav --out ./ref \
    --scan 11.5-19.5 22.0-30.0 27.0-35.0

# 2. 改 segments.json（照 segments.example.json 的格式），然后生成
python3 run_auk.py --segments segments.json --ref ./ref/11_5-19_5.wav \
    --outdir ./out --variant flash
```

产物：`out/auk_flash.wav`（整条）+ `out/flash_report.json`（每段的实际耗时与总 RTF）。

## 参数怎么定

| 参数 | 怎么定 |
|---|---|
| `--variant` | 默认 `flash`。4 步定长，快。要质量再换 `base`，慢约 4 倍 |
| `--bits` | **用 8，别用 4。** 官方结论：8-bit 与 fp32 不可区分；4-bit 下英文还行，**中文发音会退化** |
| `--nfe` | 只对 base 有效。Flash 分支根本不读这个参数，没有可调旋钮 |
| `--seed` | 同 seed 逐比特可复现。换 seed 结果有细微差别，身份指标基本不动 |
| `gen_seconds` | 目标时长。约 0.19 秒/字，也是最主要的语速旋钮 |

## 指令模板不要改

零样本 TTS 用的模板就这一条：

```
Say the following with the same voice: "{text}"
```

**换任何一个字都会出问题。** 试过把「像平时聊天那样说」加进去，结果是模型把指令原文念了出来，或者直接把参考音频里的内容复读一遍。官方 Cookbook 给的是各任务的固定模板，照抄即可。

## 实测读数

本机 M2 Max 96GB，macOS 27，MLX 后端，8-bit。178 字切 4 段：

| 档位 | 生成总耗时 | 音频时长 | RTF |
|---|---|---|---|
| Flash 4 步 | 41.3 s | 32.86 s | **1.26** |
| base 32 步 | 150.4 s | 32.86 s | **4.58** |

音色相似度（与参考录音对比的谱包络余弦）：Flash **0.992**，base 0.983。

官方在 M4 Pro 48GB 上的数据是 Flash RTF 0.6 到 1.8、base 32 步 RTF 4.2 到 7.3。官方自己也提醒这批数字波动大，同配置重复跑能差两倍以上，取区间下端。

## 我们试过但没用上的

**谱形补偿。** Flash 出来的声音比参考略显闷一点，就写了个按参考谱形做补偿的后处理。客观指标确实朝参考靠了（相似度 0.992 到 0.999），但做了一次不带标记的盲测，结果是原样更自然。**指标抬头不等于听感变好**，所以这条路没进默认流程。

**用电话录音当参考。** 想拿真聊天的语气，试过用一段电话录音当参考。语气确实对了（句子长度中位数从 0.49 秒变成 2.12 秒，像人在说话了），但手机免提加二次转录那几道损失，把高频磨没了，产物一听就是电话音色。**一条坏通道的参考，会把音质问题直接烧进产物，补不回来。** 想同时要「说话的感觉」和「好音质」，只能用同一套好设备重录一条放松状态的口语。

## 版权与许可

- 本仓库的脚本：MIT
- AuK 权重与代码：见 [官方仓库](https://github.com/Tencent-Hunyuan/AuK)。**AuK 主干是 MIT，编码器 `Qwen2.5-Omni-3B` 是独立许可**，商用前自己核一遍
- 本仓库不包含任何人的声音素材，也没有任何音频样本

生成的声音只应用于你自己有权利克隆的声音。拿别人的声音去生成内容之前，先取得同意。
