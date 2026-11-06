# EditableDesignBench

[数据集](https://huggingface.co/datasets/yomi-chen/EditableDesignBench) · [模型与进阶用法](docs/advanced.md) · [TextEditBench](https://github.com/CSU-JPG/TextEditBench)

面向设计图片的编辑评测集，共 **400 个任务**。模型输入为原图和自然语言指令；目标图和 mask 仅用于评测。

| 轨道 | 任务数 | 编辑操作 |
|---|---:|---|
| 文字（默认） | 200 | 删除、插入、替换、移动、缩放、改色 |
| 图片资产 | 200 | 删除、插入、替换、移动、缩放 |

文字集保留 200 个原始设计，每个设计 1 条任务，六种操作各 33–34 条。

下面先用 Qwen-Image-2.1 跑通两轨各 20 例，再计算像素指标和 MLLM 分数。已有模型输出时可直接[评分自己的图片](#已有图片)。

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

自动下载、解压到 `data/EditableDesignBench-v3/`，其中根目录是文字集，`assets/` 是资产集。无需登录。新版使用独立目录，保留已有 400 条文字任务及其旧实验；更新当前目录的数据时添加 `--update`。

## 3. 生成图片

```bash
python -m pip install -r requirements-generation.txt
python scripts/bench.py download --models qwen-image21

# 文字集 20 例
python scripts/bench.py generate --models qwen-image21 --sample 20 --output-root outputs/text20

# 资产集 20 例
python scripts/bench.py generate --models qwen-image21 --track asset --sample 20 --output-root outputs/asset20
```

将 `--sample 20` 换成 `--all` 即可运行所选轨道的全部 200 例。默认抽样种子为 42，按操作分层并优先选择不同原始设计。

替换 `--models` 即可使用其他[内置模型或自定义模型](docs/advanced.md#模型选择)；多个模型用逗号分隔。权重保存在 `models/`。受限模型需先在 Hugging Face 接受访问条件，再运行 `hf auth login`。

## 4. 评分

设置支持图片输入的 OpenAI 兼容评委 API：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_URL="https://your-provider.example/v1"
export OPENAI_MODEL="your-vision-model"

python scripts/bench.py evaluate --runs-root outputs/text20 --workers 10 --device cpu
python scripts/bench.py evaluate --runs-root outputs/asset20 --workers 10 --device cpu
```

评分自动读取实验中的模型、任务名单和轨道。**只计算像素指标时，跳过 API 配置并在命令后添加 `--pixel-only`。** LPIPS 首次运行会自动下载感知网络权重，需要联网。

终端显示进度；重复执行同一命令会复用成功分数，仅补算失败或未完成项。API 错误较多时将 `--workers 10` 改为 `--workers 2` 重试。

### 已有图片

将输出保存为原图尺寸的 RGB PNG，按任务 ID 命名，如 `predictions/MyModel/000_delete.png`。安装基础依赖、下载数据并配置 API 后运行：

```bash
python scripts/bench.py evaluate --predictions predictions/MyModel --workers 10 --device cpu
```

默认评分文字集全部 200 例；资产集添加 `--track asset`，子集用 `--ids ID1,ID2` 指定。结果保存在 `results/MyModel/`，也支持 `--pixel-only`。

## 查看结果

生成实验的结果位于 `outputs/text20/` 或 `outputs/asset20/`：

| 文件 | 内容 |
|---|---|
| `<model>/pixel_scores.json` | MSE、PSNR、SSIM、LPIPS |
| `<model>/mllm_scores.json` | 每例、每维度的 MLLM 分数和错误 |
| `<model>/comparison.png` | 原图、目标图与模型结果预览 |
| `reports/comparison.csv` | 模型对比表 |

像素指标复用 TextEditBench 的实现。MLLM 每例四次独立请求，每维 0–5 分，使用 `temperature=0`、`max_tokens=500`；20 例共 80 次请求，不含重试。文字轨默认用严格版 prompt 评 IF / TA / VC / LP，资产轨评 IF / OA / VC / LP，并报告四维均 ≥ 4 的联合合格率。两轨分开汇报；资产评委只接收原图和模型结果两张整图，不裁局部，两轨评委默认均不接收目标图。

资产评分逐维记录证据。分数与证据标签矛盾、或不同维度观察不一致时，仅标记供复核，仍接收原始分数并正常统计，不改分、不因此重试。联合合格率为主要成功指标，四维分数用于分析错误。资产评分是本项目扩展协议，具体规则见[进阶用法](docs/advanced.md)。

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
