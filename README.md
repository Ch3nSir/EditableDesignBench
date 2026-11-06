# EditableDesignBench

[数据集](https://huggingface.co/datasets/yomi-chen/EditableDesignBench) · [模型与进阶用法](docs/advanced.md) · [TextEditBench](https://github.com/CSU-JPG/TextEditBench)

面向设计图片的编辑评测集，共 **400 个任务**。模型输入为原图和自然语言指令；目标图和 mask 仅用于评测。

| 轨道 | 任务数 | 编辑操作 |
|---|---:|---|
| 文字（默认） | 200 | 删除、插入、替换、移动、缩放、改色 |
| 图片资产 | 200 | 删除、插入、替换、移动、缩放 |

文字集保留 200 个原始设计，每个设计 1 条任务，六种操作各 33–34 条。

下面用 Qwen-Image-2.1 生成并评测完整 400 个任务，两条轨道分别运行、分别汇报。已有模型输出时可直接[评分自己的图片](#已有图片)。

## 1. 安装环境

```bash
git clone https://github.com/Ch3nSir/EditableDesignBench.git
cd EditableDesignBench
conda create -n editabledesignbench python=3.12 -y
conda activate editabledesignbench
python -m pip install -r requirements.txt
```

使用 Linux、Python 3.12。下载与评分可在 CPU 上运行；本地模型生成需要 NVIDIA GPU 和支持 CUDA 的 PyTorch，显存需求随模型和图片尺寸变化。已有 PyTorch 时，安装过程会保留满足版本范围的版本。

| 依赖文件 | 用途 |
|---|---|
| `requirements.txt` | 基础依赖：下载数据、像素指标、MLLM 评分 |
| `requirements-generation.txt` | 本地模型生成，包含全部基础依赖 |
| `requirements.lock.txt` | 可选：还原已验证的完整环境（Python 3.12 / PyTorch 2.8 / CUDA 12.8） |

常规安装使用版本范围，无需安装锁定文件。Diffusers 保留仓库自带构建，以提供本项目使用的新版管线。CPU / CUDA 安装选择及精确复现环境的命令见[环境说明](docs/advanced.md#环境与依赖)。

## 2. 下载数据

```bash
python scripts/download_dataset.py
```

默认下载固定发布版 `v3.0.0`，解压到 `data/EditableDesignBench-v3/`：根目录是 200 条文字任务，`assets/` 是 200 条资产任务。无需登录。

## 3. 生成图片

生成前需安装与显卡驱动匹配的 CUDA 版 PyTorch，安装命令见[环境说明](docs/advanced.md#环境与依赖)。

```bash
python -m pip install -r requirements-generation.txt
python scripts/bench.py download --models qwen-image21

# 文字集：全部 200 例
python scripts/bench.py generate --models qwen-image21 --all --output-root outputs/text

# 资产集：全部 200 例
python scripts/bench.py generate --models qwen-image21 --track asset --all --output-root outputs/assets
```

以上两条生成命令合计 400 例。生成需在当前进程能访问 CUDA GPU 的环境中执行；评分可以使用 CPU。

替换 `--models` 即可使用其他[内置模型或自定义模型](docs/advanced.md#模型选择)；多个模型用逗号分隔。权重保存在 `models/`。受限模型需先在 Hugging Face 接受访问条件，再运行 `hf auth login`。

## 4. 评分

设置支持图片输入的 OpenAI 兼容评委 API：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_URL="https://your-provider.example/v1"
export OPENAI_MODEL="your-vision-model"

python scripts/bench.py evaluate --runs-root outputs/text --workers 10 --device cpu
python scripts/bench.py evaluate --runs-root outputs/assets --workers 10 --device cpu
```

评分自动读取实验中的模型、任务名单和轨道。**只计算像素指标时，跳过 API 配置并在命令后添加 `--pixel-only`。** LPIPS 首次运行会自动下载感知网络权重，需要联网。

终端显示进度；重复执行同一命令会复用成功分数，仅补算失败或未完成项。API 错误较多时将 `--workers 10` 改为 `--workers 2` 重试。

### 已有图片

将输出保存为原图尺寸的 RGB 或 RGBA PNG，按任务 ID 命名，如 `predictions/MyModel/000_delete.png`。RGBA 直接读取原始 RGB 通道，不合成背景、不修改文件。安装基础依赖、下载数据并配置 API 后运行：

```bash
python scripts/bench.py evaluate --predictions predictions/MyModel --workers 10 --device cpu
```

以上评分文字集全部 200 例。再对资产集输出运行一次，合计 400 例：

```bash
python scripts/bench.py evaluate --predictions predictions/MyModel_assets --track asset --workers 10 --device cpu
```

结果分别保存在 `results/MyModel/`、`results/MyModel_assets/`，也支持 `--pixel-only`。

## 查看结果

生成实验的结果位于 `outputs/text/` 或 `outputs/assets/`：

| 文件 | 内容 |
|---|---|
| `<model>/pixel_scores.json` | MSE、PSNR、SSIM、LPIPS |
| `<model>/mllm_scores.json` | 每例、每维度的 MLLM 分数和错误 |
| `<model>/report.json` | 经过校验的指标、联合合格率和覆盖率 |
| `<model>/comparison.png` | 原图、目标图与模型结果预览 |
| `reports/comparison.csv` | 模型对比表 |

像素指标复用 TextEditBench 的实现。MLLM 每例四次独立请求，每维 0–5 分，使用 `temperature=0`、`max_tokens=500`；每模型完整 400 例共 1,600 次请求，不含重试。文字轨默认用严格版 prompt 评 IF / TA / VC / LP，资产轨评 IF / OA / VC / LP，并报告四维均 ≥ 4 的联合合格率。两轨分开汇报；资产评委只接收原图和模型结果两张整图，不裁局部，两轨评委默认均不接收目标图。

合法的维度分数正常参与统计；缺少辅助说明、证据标签矛盾等只作记录，不改分、不因此重试。某个模型失败不清空其他模型的结果；不同实验协议的分数保留，但会标明不可直接比较。联合合格率为主要成功指标，四维分数用于分析错误。资产评分是本项目扩展协议，具体规则见[进阶用法](docs/advanced.md)。

## 许可证与引用

代码采用 MIT，数据集采用 [CC BY-NC 4.0](https://huggingface.co/datasets/yomi-chen/EditableDesignBench)。第三方组件见 [NOTICE.md](NOTICE.md)。

```bibtex
@misc{editabledesignbench2026,
  title = {EditableDesignBench},
  author = {{yomi-chen}},
  year = {2026},
  url = {https://huggingface.co/datasets/yomi-chen/EditableDesignBench}
}
```
