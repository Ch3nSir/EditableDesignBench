# 进阶用法

常规安装、生成和评分见 [README](../README.md)。以下内容按需使用。

## 模型选择

`download` 和 `generate` 的 `--models all` 是默认组合 `flux3-klein,qwen-image21,qwen-image21-turbo`。评分和汇总中的 `all` 表示实验目录里记录的全部模型。

生成可以用 `--cpu-offload` 节省显存；`--preflight` 仅检查权重和调用参数，不加载模型。FLUX 3 的本地官方提示词扩写器随模型下载，不调用评委 API；评分使用原始指令。Turbo 使用权重自带的八步采样日程。

### 接入新的 Diffusers 模型

内置模型不需要配置文件。接入新模型时，将自己的配置保存为 `config/models.json`；字段示例见 [models.example.json](../config/models.example.json)。

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
  --device cuda

python scripts/bench.py compare --runs-root outputs/custom_smoke --models all
```

配置扩展适用于已安装 Diffusers 中以 `prompt`、单张 `image`、`width`、`height`、`num_inference_steps` 和 `generator` 调用，并返回 `result.images` 的图片编辑管线。管线还需要暴露 `vae_scale_factor`，其尺寸倍数须兼容当前补边规则。配置不能重写原图、指令、尺寸、步数或随机数生成器。特殊预处理、自定义返回格式或 API 模型需要自己的推理脚本，并按 README 中的 `--predictions` 方式提交结果。配置模板中的模型只是接入示例，不代表任意模型都能直接替换仓库 ID 运行。

新增模型可以与其他模型放到同一个实验目录、使用同一组 case；每个模型保存在独立子目录。评分和汇总会自动读取记录的模型 ID，无需再次传入模型配置。全量运行时将 `--ids 000_delete` 换成 `--all`。

## API 与请求参数

README 中的环境变量足以使用 GPT-6-Luna，脚本自动选择 `reasoning_chat`：`reasoning_effort=low`、`max_completion_tokens=4096`、`store=false`，不发送 `temperature`。四个维度分别独立请求。

如需修改请求参数，可以创建自己的 JSON，通过 `evaluate --config path/to/judge.json` 传入。字段示例见 [judge.example.json](../config/judge.example.json)。显式请求配置优先于自动配置。

其他评委默认沿用 TextEditBench 的 `temperature=0`、`max_tokens=500`；推理模型需在 JSON 中显式设置 `request_profile=reasoning_chat`。比较模型时应使用同一个评委和请求参数。

也可以将 `OPENAI_API_KEY`、`OPENAI_API_URL`、`OPENAI_MODEL` 写入仓库根目录的 `.env`，每行 `变量名=值`。脚本自动读取，已导出的环境变量优先。不要提交自己的密钥。

## 路径与输出尺寸

默认数据在 `data/EditableDesignBench/`，权重在 `models/`；可用 `EDITABLEDESIGNBENCH_DATASET_ROOT`、`EDITABLEDESIGNBENCH_MODEL_ROOT` 环境变量修改。

生成器只读取原图和指令，不读取目标图或 mask。原图只在右侧和底部补边到 32 的倍数。模型原始返回图完整保留，评测时在内存中去掉额外补边。直接提交的预测图须为原生尺寸 RGB PNG，不能将带补边图片当作原生尺寸图片提交。

## 子集、续跑与汇总

首次测试可使用 `generate --ids 000_delete`。生成后评分会自动使用记录的同一组任务。为其他模型图片评分时，用 `evaluate --predictions predictions/MyModel --ids 000_delete` 指定子集，默认评测全部 400 例。

同一实验可以续跑，成功图片和匹配的评分会复用；修改任务集合、权重或生成参数后使用新的输出目录。

```bash
# 重新汇总已生成实验中的模型评分
python scripts/bench.py compare --runs-root outputs/run1
```

仅像素实验重新汇总时加 `--pixel-only`。汇总核对数据、任务集和评分协议；失败项保留错误与覆盖率，未覆盖完整任务集的指标均分为 `null`，不补零、不合成总分。

已有图片的评分会在 `results/<模型文件夹名>/` 输出 `pixel_scores.json` 和 `mllm_scores.json`；该入口直接评分，不创建内置生成器的多模型实验目录。可用 `--output-dir` 修改结果目录，或用 `--dry-run` 检查请求输入而不调用 API。

## 方法与验证

像素公式复用随仓库提供的 TextEditBench 评分源码：MSE/PSNR 使用官方配准流程，SSIM/LPIPS 使用未配准图片；每例独立计算后取逐例算术平均。评测与预览只处理内存中的图像数组，保留原始模型输出。

本地已验证三模型的普通尺寸及最大尺寸单例；FLUX 3 的 20 例测试完成了全部四项像素指标和 80 次 GPT-6-Luna 评分。测试硬件包括 H200 和 96 GB RTX PRO 6000 Blackwell。

离线回归检查：

```bash
python -m unittest discover -s scripts -p '*test*.py'
```
