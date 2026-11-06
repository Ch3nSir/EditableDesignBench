# 🎨 EditableDesignBench

[🤗 数据集](https://huggingface.co/datasets/yomi-chen/EditableDesignBench) · [📖 进阶用法与模型列表](docs/advanced.md)

面向设计图片的编辑评测集，共 **400 个任务**。模型接收原图和自然语言指令；目标图与 mask 仅用于评测。

| 轨道 | 任务数 | 编辑操作 |
|---|---:|---|
| 文字 | 200 | 删除、插入、替换、移动、缩放、改色 |
| 图片资产 | 200 | 删除、插入、替换、移动、缩放 |

## 🚀 快速开始

以下以 **Qwen-Image-2.1** 为例，完成全部 400 个任务的生成与评分。

### 1. 安装环境

```bash
git clone https://github.com/Ch3nSir/EditableDesignBench.git
cd EditableDesignBench
conda create -n editabledesignbench python=3.12 -y
conda activate editabledesignbench
python -m pip install -r requirements-generation.txt
```

本地生成需要 Linux、CUDA GPU 和 CUDA 版 PyTorch，评分支持 CPU。[CUDA 安装与锁定环境](docs/advanced.md#环境与依赖)见进阶文档。生成依赖已包含评测依赖；**只评测已有图片时，改装 `requirements.txt`**。`requirements.lock.txt` 用于精确还原已验证环境，可选。

### 2. 下载数据

```bash
python scripts/download_dataset.py
```

自动下载发布版 `v3.0.0` 到 `data/EditableDesignBench-v3/`，包含文字和资产两轨，无需登录。

### 3. 生成图片

```bash
python scripts/bench.py download --models qwen-image21

# 文字：全部 200 例
python scripts/bench.py generate --models qwen-image21 --all --output-root outputs/text

# 资产：全部 200 例
python scripts/bench.py generate --models qwen-image21 --track asset --all --output-root outputs/assets
```

替换 `--models` 即可测试[其他内置模型或自定义模型](docs/advanced.md#模型选择)，多个模型用逗号分隔。受限权重需先在 Hugging Face 接受访问条件，再运行 `hf auth login`。

### 4. 配置评委并评分

使用支持图片输入的 OpenAI 兼容 API：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_URL="https://your-provider.example/v1"
export OPENAI_MODEL="your-vision-model"

python scripts/bench.py evaluate --runs-root outputs/text --workers 10 --device cpu
python scripts/bench.py evaluate --runs-root outputs/assets --workers 10 --device cpu
```

每模型全量评测共 **1,600 次评委请求**，不含重试。仅计算像素指标时，无需 API，添加 `--pixel-only` 即可。LPIPS 首次运行需联网下载权重。

重复执行同一命令会自动续跑，只补失败或未完成项。接口错误较多时，可将 `--workers 10` 改为 `--workers 2`。

## 🧩 评测已有图片

完成环境安装、数据下载和 API 配置后，将原图尺寸的 RGB/RGBA PNG 按任务 ID 命名，如 `predictions/MyModel_text/000_delete.png`：

```bash
# 文字 200 例
python scripts/bench.py evaluate --predictions predictions/MyModel_text --workers 10 --device cpu

# 资产 200 例
python scripts/bench.py evaluate --predictions predictions/MyModel_assets --track asset --workers 10 --device cpu
```

结果分别保存在 `results/MyModel_text/` 和 `results/MyModel_assets/`。同样支持 `--pixel-only`。

## 📊 查看结果

两条轨道单独汇报，生成实验的输出位于 `outputs/text/` 和 `outputs/assets/`：

| 文件 | 内容 |
|---|---|
| `<model>/pixel_scores.json` | MSE、PSNR、SSIM、LPIPS |
| `<model>/mllm_scores.json` | 每例 MLLM 分数与诊断 |
| `<model>/report.json` | 完整性、覆盖率及联合合格率 |
| `<model>/comparison.png` | 图片对比预览 |
| `reports/comparison.csv` | 模型对比表 |

MLLM 每维 0–5 分；**联合合格率**要求四个维度均 ≥ 4。评分维度、提示词及恢复机制见[评测说明](docs/advanced.md#文字与资产轨)。

## 📝 引用与许可证

代码采用 MIT，数据集采用 [CC BY-NC 4.0](https://huggingface.co/datasets/yomi-chen/EditableDesignBench)。评测实现参考 [TextEditBench](https://github.com/CSU-JPG/TextEditBench)，第三方组件见 [NOTICE](NOTICE.md)。

```bibtex
@misc{editabledesignbench2026,
  title = {EditableDesignBench},
  author = {{yomi-chen}},
  year = {2026},
  url = {https://huggingface.co/datasets/yomi-chen/EditableDesignBench}
}
```
