# EditableDesignBench

[数据集](https://huggingface.co/datasets/yomi-chen/EditableDesignBench) · [进阶用法](docs/advanced.md) · [TextEditBench](https://github.com/CSU-JPG/TextEditBench)

图片文字编辑评测集：**200 个原始设计、400 个编辑任务**，覆盖删除、插入、替换、移动、缩放和改色。模型输入为原图和自然语言指令；目标图和 mask 仅用于评分。

## 1. 安装

```bash
git clone https://github.com/Ch3nSir/EditableDesignBench.git
cd EditableDesignBench
conda env create -f environment.yml
conda activate editabledesignbench
pip install -r requirements.lock.txt
```

使用 Linux、Python 3.12 和支持 CUDA 的 GPU。依赖文件已固定所需版本。

## 2. 下载数据

```bash
python scripts/download_dataset.py
```

自动下载并解压到 `data/EditableDesignBench/`，无需登录。

## 3. 设置评委 API

在当前终端设置自己的 OpenAI 兼容视觉 API：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-6-luna"
```

使用代理时修改 `OPENAI_API_URL`。评委请求沿用 TextEditBench 的 `temperature=0`、`max_tokens=500`。无需准备 JSON 配置文件；只计算像素指标时可跳过此步。

## 4. 生成图片

选择一个模型，下载权重并生成 400 张结果：

```bash
python scripts/bench.py download --models flux3-klein
python scripts/bench.py generate --models flux3-klein --all --output-root outputs/run1
```

内置模型如下；替换命令中的模型 ID 即可。多个模型用逗号分隔，例如 `--models flux3-klein,qwen-image21`，下载和生成时使用同一组 ID。

| 模型 ID | 模型 |
|---|---|
| `flux3-klein` | [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) |
| `qwen-image21` | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| `qwen-image21-turbo` | [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) |
| `qwen-edit-2511` | [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| `flux2-klein-9b` | [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| `flux2-dev` | [FLUX.2 Dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) |

前三种已做真实 GPU 测试，后三种已实现适配，尚未完成真实 GPU 生成验证。权重存放在 `models/`；如下载提示需要授权，在模型页面接受许可后运行 `hf auth login`。

先试一张时，把 `--all` 换成 `--ids 000_delete`，并使用单独的输出目录。已有其他模型生成的图片，可跳过这一步，直接使用下一节的图片评分命令。新增模型配置见[进阶用法](docs/advanced.md)。

## 5. 评分

使用 10 并发评测上一步生成的所有模型，并自动汇总：

```bash
python scripts/bench.py evaluate --runs-root outputs/run1 --workers 10 --device cuda
```

无需再次指定模型或任务。各模型的像素分数、MLLM 分数和图片预览在 `outputs/run1/<模型ID>/`，对比表在 `outputs/run1/reports/comparison.csv`。

若已有自己生成的图片，将其保存为与原图同尺寸的 RGB PNG，以任务 ID 命名，例如 `predictions/MyModel/000_delete.png`。一条命令完成评分：

```bash
python scripts/bench.py evaluate --predictions predictions/MyModel --workers 10 --device cuda
```

分数保存在 `results/MyModel/`。只测部分图片时添加 `--ids 000_delete,000_insert`，默认评测全部 400 例。

`--workers` 控制每个评测进程并发处理的 case 数；不指定时默认 **5**，以上示例使用 **10**。终端自动显示维度进度、完成 case 数、失败数、缓存命中数、用时和预计剩余时间；添加 `--no-progress` 可关闭显示。

**重复执行原命令即可断点续跑**：像素指标逐项复用成功值，MLLM 逐维度复用成功分数，只补算失败或未完成项。终端会显示 `cached` / `reused_metrics`；需要全部重算时才添加 `--no-resume`。

两种评分命令都可以添加 `--pixel-only` 跳过 API；像素评分没有 GPU 时，将 `--device cuda` 改为 `--device cpu`。

像素指标为 **MSE、PSNR、SSIM、LPIPS**，沿用 TextEditBench 的实现。MLLM 按 **IF、TA、VC、LP** 四个维度分别评分，每个任务四次独立请求；一个模型的完整 400 例共 1,600 次请求，不含重试。默认输入原图、结果图和指令，不提供目标图，不计算 SE。

自定义模型、请求参数、续跑和指标细节见[进阶用法](docs/advanced.md)。

## 许可证与引用

代码采用 MIT，数据集采用 [CC BY-NC 4.0](https://huggingface.co/datasets/yomi-chen/EditableDesignBench)。第三方组件许可证见 [NOTICE.md](NOTICE.md)。

```bibtex
@misc{editabledesignbench2026,
  title = {EditableDesignBench},
  author = {{yomi-chen}},
  year = {2026},
  url = {https://huggingface.co/datasets/yomi-chen/EditableDesignBench}
}
```
