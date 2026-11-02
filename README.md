# EditableDesignBench

[数据集](https://huggingface.co/datasets/yomi-chen/EditableDesignBench) · [进阶用法](docs/advanced.md) · [TextEditBench](https://github.com/CSU-JPG/TextEditBench)

图像编辑评测集，共 **600 个任务**，分为文字与图片资产两条轨道。模型输入为原图和自然语言指令；目标图和 mask 仅用于评测。

| 轨道 | 任务数 | 操作 | MLLM 维度 |
|---|---:|---|---|
| 文字（默认） | 400 | 删除、插入、替换、移动、缩放、改色 | IF / TA / VC / LP |
| 图片资产 | 200 | 删除、插入、替换、移动、缩放 | IF / OA / VC / LP |

文字任务来自 200 个原始设计，资产任务来自其中 34 个设计的 40 个资产单元。两轨分别运行和汇报分数。

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

一次下载两条轨道，自动解压到 `data/EditableDesignBench/`，无需登录。文字数据位于根目录，资产数据位于 `assets/`。

已下载过旧版时，运行 `python scripts/download_dataset.py --update`；新版通过校验后才替换旧目录，并保留旧目录备份。

## 3. 设置评委 API

在当前终端设置自己的 OpenAI 兼容视觉 API：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_URL="https://your-provider.example/v1"
export OPENAI_MODEL="your-vision-model"
```

填入支持图片输入的评委 API 地址和模型 ID。评委请求使用 `temperature=0`、`max_tokens=500`。无需准备 JSON 配置文件；只计算像素指标时可跳过此步。

## 4. 生成图片

选择一个模型，下载权重，先抽样试跑：

```bash
python scripts/bench.py download --models qwen-image21

# 文字集抽 20 例
python scripts/bench.py generate --models qwen-image21 --sample 20 --output-root outputs/text20

# 资产集抽 20 例
python scripts/bench.py generate --models qwen-image21 --track asset --sample 20 --output-root outputs/asset20
```

将 `--sample 20` 换成 `--all` 即可运行整条轨道。抽样按操作分层、优先选择不同原始设计，默认种子为 42；任务名单写入实验目录，评分自动沿用同一名单。

内置模型如下；替换命令中的模型 ID 即可。多个模型用逗号分隔，例如 `--models flux3-klein,qwen-image21`，下载和生成时使用同一组 ID。

| 模型 ID | 模型 |
|---|---|
| `flux3-klein` | [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) |
| `qwen-image21` | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| `qwen-image21-turbo` | [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) |
| `qwen-edit-2511` | [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| `flux2-klein-9b` | [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| `flux2-dev` | [FLUX.2 Dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| `sensenova-u1.5-8b-mot` | [SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) |
| `fibo2-edit-turbo` | [Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) |
| `joyai-image-edit-plus` | [JoyAI-Image-Edit-Plus](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers) |

权重存放在 `models/`；如下载提示需要授权，在模型页面接受许可后运行 `hf auth login`。SenseNova 的额外安装、各模型参数和自定义模型接入见[进阶用法](docs/advanced.md)。

模型原始输出完整保留。RGBA 输出直接提取原始 RGB 三通道供评测，不叠加背景；Alpha 仅记录统计，不影响评测资格。

指定单个任务时使用 `--ids 000_delete`；资产任务例如 `--track asset --ids asset_000_a00_delete`。已有其他模型生成的图片，可直接使用下一节的图片评分命令。

## 5. 评分

使用 10 并发评测上一步生成的所有模型，并自动汇总：

```bash
python scripts/bench.py evaluate --runs-root outputs/text20 --workers 10 --device cpu
python scripts/bench.py evaluate --runs-root outputs/asset20 --workers 10 --device cpu
```

无需再次指定模型、任务或轨道。每个实验中的模型分数和 `comparison.png` 预览位于模型子目录，对比表位于 `reports/comparison.csv`。VS Code 可直接打开 PNG 预览。有可用 GPU 时可把 `--device cpu` 改为 `--device cuda`。

若已有自己生成的图片，将其保存为与原图同尺寸的 RGB PNG，以任务 ID 命名，例如 `predictions/MyModel/000_delete.png`。一条命令完成评分：

```bash
python scripts/bench.py evaluate --predictions predictions/MyModel --workers 10 --device cuda
```

分数保存在 `results/MyModel/`。资产图片评分添加 `--track asset`。只测部分图片时添加 `--ids`；默认评测所选轨道的全部任务。

`--workers` 控制每个评测进程并发处理的 case 数；不指定时默认 **5**，以上示例使用 **10**。终端自动显示维度进度、完成 case 数、失败数、缓存命中数、用时和预计剩余时间；添加 `--no-progress` 可关闭显示。

**重复执行原命令即可断点续跑**：像素指标逐项复用成功值，MLLM 逐维度复用成功分数，只补算失败或未完成项。API 失败较多时，可将原命令的 `--workers 10` 改为 `--workers 2` 补评。终端会显示 `cached` / `reused_metrics`；需要全部重算时才添加 `--no-resume`。

两种评分命令都可以添加 `--pixel-only` 跳过 API。

像素指标为 **MSE、PSNR、SSIM、LPIPS**，复用 TextEditBench 的实现。文字轨使用源/目标编辑区域并集；资产轨使用根据源图与指令定义的 `allowed_edit_mask.png`，不要求复刻唯一参考素材。

MLLM 每个任务四次独立请求，每维 0–5 分。文字轨为 IF、TA、VC、LP；资产轨为 IF、OA（资产状态正确性）、VC、LP，并提供整图和同坐标局部放大。两轨均默认不向评委提供目标图。资产轨另报四维均达到 4 分的联合合格率，以全部所选任务为分母；模型缺失输出计不合格，API 错误保留待评。资产协议是本项目扩展，不能视为 TextEditBench 官方资产评分。

不含重试时，每轨 20 例为 80 次请求；完整文字轨为 1,600 次，完整资产轨为 800 次。

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
