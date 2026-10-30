# EditableDesignBench

[数据集 / Dataset](https://huggingface.co/datasets/yomi-chen/EditableDesignBench) · [TextEditBench](https://github.com/CSU-JPG/TextEditBench)

EditableDesignBench 用于评测图片中的文字编辑能力，包含 **200 个原始设计、400 个编辑任务**，覆盖删除、插入、替换、移动、缩放和改色。

本仓库提供图片生成适配、像素指标、MLLM 评分、图片预览和多模型汇总。默认模型输入只有**原图和自然语言指令**，目标图与评测 mask 仅用于评分。

## 1. 安装环境

```bash
git clone https://github.com/Ch3nSir/EditableDesignBench.git
cd EditableDesignBench

conda env create -f environment.yml
conda activate editabledesignbench
pip install -r requirements.lock.txt
python scripts/bench.py check-env --prepare-metrics
```

环境使用 Linux、Python 3.12、CUDA 12.8 版 PyTorch。依赖文件包含已经验证的 Diffusers 版本，安装时直接使用提供的版本。`--prepare-metrics` 会准备 LPIPS 所需的小型预训练权重，不加载图片生成模型。

## 2. 下载数据集

```bash
python scripts/download_dataset.py
```

从公开 Hugging Face 数据集下载并解压到 `data/EditableDesignBench/`，无需登录。文件结构如下：

```text
data/EditableDesignBench/
├── tasks.jsonl
├── references.jsonl
├── summary.json
└── cases/<id>/
    ├── source.png
    ├── target.png
    ├── mask.png
    ├── target_region_mask.png
    ├── eval_mask.png
    ├── task.json
    └── annotation.json
```

`tasks.jsonl` 是模型输入清单，`references.jsonl` 是评测参考清单。数据使用原生图片尺寸；mask 是矩形框的并集，白色表示编辑区域，黑色表示应保留区域。

## 3. 配置评委 API

```bash
cp .env.example .env
cp config/judge.example.json config/judge.json
```

编辑 `.env`，填写自己的 OpenAI 兼容视觉 API：

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-6-luna
```

使用代理时，将 `OPENAI_API_URL` 改成代理的 API 基地址，例如 `https://your-provider.example/v1`。脚本自动读取仓库根目录的 `.env`；已经导出的系统环境变量优先。

示例 `config/judge.json` 使用以下请求配置：

```json
{
  "api_key_env": "OPENAI_API_KEY",
  "request_profile": "reasoning_chat",
  "reasoning_effort": "low",
  "max_completion_tokens": 4096,
  "store": false,
  "timeout": 120,
  "max_retries": 2
}
```

默认评委为 **GPT-6-Luna**。每个 case 分别请求 IF、TA、VC、LP，共四次独立请求，沿用 TextEditBench 的评分提示词；每次输入原图、模型结果图和原始编辑指令，返回 0–5 分及理由。默认不提供目标图。这批数据没有 `knowledge_prompt`，因此不计算 SE。

`reasoning_chat` 使用 `max_completion_tokens`，不发送 `temperature`。如果使用原版 GPT-4o 等旧参数接口，将 `request_profile` 改为 `texteditbench`，该模式使用 `temperature=0`、`max_tokens=500`。同一次模型比较需要使用同一评委和请求配置。

## 4. 生成模型结果

生成和评分支持多个模型。内置六种模型配置，也可以通过 JSON 增加兼容的 Diffusers 模型；使用其他推理框架或 API 时，可以提交生成好的图片，见下方“评测已有图片”。

### 下载模型

```bash
python scripts/bench.py download --models all
```

权重默认放在 `models/`。`--models all` 是默认三模型组合：`flux3-klein,qwen-image21,qwen-image21-turbo`。完整的内置配置如下，可以用 ID 单独选择或用逗号组合：

| 模型 ID | 官方仓库 | 步数 |
|---|---|---:|
| `flux3-klein` | [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | 16 |
| `qwen-image21` | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 40 |
| `qwen-image21-turbo` | [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | 8 |
| `qwen-edit-2511` | [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | 40 |
| `flux2-klein-9b` | [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | 4 |
| `flux2-dev` | [FLUX.2 Dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 50 |

表中是本仓库的采样配置。后三种模型已有调用参数、依赖和权重完整性检查；当前版本尚未完成它们在本数据集上的真实 GPU 生成测试。先用一例确认显存和输出尺寸再运行全量。查看当前全部模型 ID：

```bash
python scripts/bench.py list-models
```

如模型仓库要求授权，先在对应页面接受许可，再运行 `hf auth login`。FLUX 3 的官方本地提示词扩写器会一并下载；它是模型组件，不调用评委 API。扩写后的提示词另存记录，评分始终使用原始编辑指令。Turbo 使用权重自带的 8 步采样日程。

### 先跑一例

在已经分配 GPU 的环境中运行：

```bash
python scripts/bench.py generate \
  --models all \
  --ids 000_delete \
  --output-root outputs/smoke1
```

本地模式按模型顺序运行，并在模型之间释放进程。也可以只运行一个模型，例如 `--models flux3-klein`；多个模型用逗号分隔。显存不足时可以添加 `--cpu-offload`。

默认路径可以在 `.env` 中用 `EDITABLEDESIGNBENCH_MODEL_ROOT` 和 `EDITABLEDESIGNBENCH_DATASET_ROOT` 修改。输出保留原生画布，仅在右侧和底部补边到 32 的倍数；原始返回图片完整保存，评测时只去掉额外补边。

### 选择其他内置模型

例如同时测试 Qwen-Image-Edit-2511 和 FLUX.2 Klein 9B：

```bash
python scripts/bench.py download --models qwen-edit-2511,flux2-klein-9b

python scripts/bench.py generate \
  --models qwen-edit-2511,flux2-klein-9b \
  --ids 000_delete \
  --output-root outputs/more_models

python scripts/bench.py evaluate \
  --runs-root outputs/more_models \
  --models all \
  --config config/judge.json \
  --device cuda
```

**评分和汇总时的 `--models all` 表示该实验目录里的全部模型**，包括额外模型；下载和生成时则表示上面的默认三模型组合。若要生成全部六种内置模型，下载和生成命令均使用 `--models flux3-klein,qwen-image21,qwen-image21-turbo,qwen-edit-2511,flux2-klein-9b,flux2-dev`。

### 接入新的 Diffusers 模型

无需修改源码，复制模型配置模板：

```bash
cp config/models.example.json config/models.json
```

模板用 Qwen-Image-Edit-2511 演示一个额外配置 ID `my-qwen-edit`。接入其他模型时，按该模型的推理方式修改仓库 ID、Diffusers 类名、源码位置、采样参数和完整组件目录列表。`local_dir` 是 `models/` 下独立的文件夹名，不能覆盖其他模型目录。

```bash
python scripts/bench.py list-models --model-config config/models.json
python scripts/bench.py download --models my-qwen-edit --model-config config/models.json

# 只检查本地权重、调用参数和输入，不加载模型
python scripts/bench.py generate \
  --models my-qwen-edit --model-config config/models.json \
  --ids 000_delete --output-root outputs/custom_smoke --preflight

# 配置不变时，在同一目录执行真实生成
python scripts/bench.py generate \
  --models my-qwen-edit --model-config config/models.json \
  --ids 000_delete --output-root outputs/custom_smoke

python scripts/bench.py evaluate \
  --runs-root outputs/custom_smoke --models all \
  --config config/judge.json --device cuda

python scripts/bench.py compare --runs-root outputs/custom_smoke --models all
```

配置扩展适用于已安装 Diffusers 中以 `prompt`、单张 `image`、`width`、`height`、`num_inference_steps` 和 `generator` 调用，并返回 `result.images` 的图片编辑管线。管线还需要暴露 `vae_scale_factor`，其尺寸倍数须兼容当前补边规则。配置不能重写原图、指令、尺寸、步数或随机数生成器。特殊预处理、自定义返回格式或 API 模型需要自己的推理脚本，并按“评测已有图片”提交结果。配置模板中的模型只是接入示例，不代表任意模型都能直接替换仓库 ID 运行。

新增模型可以与其他模型放到同一个实验目录、使用同一组 case；每个模型保存在独立子目录。评分和汇总会自动读取记录的模型 ID，无需再次传入模型配置。全量运行时将 `--ids 000_delete` 换成 `--all`。

## 5. 评分与汇总

### 同时计算像素指标和 MLLM 分数

```bash
python scripts/bench.py evaluate \
  --runs-root outputs/smoke1 \
  --models all \
  --config config/judge.json \
  --device cuda
```

该命令依次生成 MSE、PSNR、SSIM、LPIPS，图片对比预览，IF、TA、VC、LP，以及最终汇总。示例为三模型各一例，正常情况下共 12 次评委请求，不含重试。

如果暂时没有 API，只计算像素指标：

```bash
python scripts/bench.py evaluate \
  --runs-root outputs/smoke1 \
  --models all \
  --pixel-only \
  --device cuda
```

没有可用 GPU 时，像素评分可改为 `--device cpu`。评委请求使用 API，不占用本地 GPU。

结果文件：

```text
outputs/smoke1/
├── <model-id>/
│   ├── generation_manifest.json
│   ├── pixel_scores.json
│   ├── mllm_scores.json
│   ├── report.json
│   ├── comparison.png
│   └── comparison.html
└── reports/
    ├── comparison.csv
    └── comparison.json
```

汇总会核对数据版本、任务集合、输入图片和评分协议。失败项记录错误和覆盖率，不伪造为零分；某个指标没有覆盖完整任务集时，其均分为 `null`。各指标单独报告，不合成任意总分。

仅重新汇总已有评分：

```bash
python scripts/bench.py compare --runs-root outputs/smoke1 --models all
```

仅像素实验重新汇总时加 `--pixel-only`。同一实验目录可以续跑，已验证的成功图片和评分会复用。改变任务集合、模型权重或生成配置时使用新的输出目录。

## 全量 400 例

小样本确认后，使用新目录运行：

```bash
python scripts/bench.py generate --models all --all --output-root outputs/full400
python scripts/bench.py evaluate --runs-root outputs/full400 --models all --config config/judge.json --device cuda
```

三模型共生成 1,200 张图片，正常情况下是 4,800 次评委请求，不含重试。只测 FLUX 3 时，将两条命令的 `--models all` 都改为 `--models flux3-klein`。

## 评测已有图片

将其他模型生成的图片保存为原生尺寸 RGB PNG，文件名对应任务 ID：

```text
predictions/MyModel/
├── 000_delete.png
├── 000_insert.png
└── ...
```

两个评测轨道分别运行：

```bash
python scripts/evaluate.py \
  --dataset-root data/EditableDesignBench \
  --predictions predictions/MyModel \
  --output results/MyModel/pixel_scores.json \
  --device cuda

python scripts/mllm_evaluate.py \
  --dataset data/EditableDesignBench \
  --predictions predictions/MyModel \
  --output results/MyModel/mllm_scores.json \
  --config config/judge.json \
  --require-scoring
```

单独测试子集时，两条命令都添加同一组 `--ids`，例如 `--ids 000_delete,000_insert`。直接提交的预测图需要与原图同尺寸；内置生成器的补边输出应通过 `generation_manifest.json` 评测，不能直接当作原生尺寸图片。

## 方法与验证

像素公式复用随仓库提供的 TextEditBench 评分源码：MSE/PSNR 使用官方配准流程，SSIM/LPIPS 使用未配准图片；每例独立计算后取逐例算术平均。评测与预览只处理内存中的图像数组，保留原始模型输出。

本地已验证三模型的普通尺寸及最大尺寸单例；FLUX 3 的 20 例测试完成了全部四项像素指标和 80 次 GPT-6-Luna 评分。测试硬件包括 H200 和 96 GB RTX PRO 6000 Blackwell。

离线回归检查：

```bash
python -m unittest discover -s scripts -p '*test*.py'
```

## 许可证与引用

本仓库代码采用 MIT 许可证，数据集采用 [CC BY-NC 4.0](https://huggingface.co/datasets/yomi-chen/EditableDesignBench)。第三方组件保留其原许可证，见 [NOTICE.md](NOTICE.md)。

```bibtex
@misc{editabledesignbench2026,
  title = {EditableDesignBench},
  author = {{yomi-chen}},
  year = {2026},
  url = {https://huggingface.co/datasets/yomi-chen/EditableDesignBench}
}
```
