# 进阶用法

常规安装、生成和评分见 [README](../README.md)。以下内容按需使用。

## 模型选择

`download` 和 `generate` 的 `--models all` 是默认组合 `flux3-klein,qwen-image21,qwen-image21-turbo`。评分和汇总中的 `all` 表示实验目录里记录的全部模型。

生成可以用 `--cpu-offload` 节省显存；`--preflight` 仅检查权重和调用参数，不加载模型。FLUX 3 的本地官方提示词扩写器随模型下载，不调用评委 API；评分使用原始指令。Turbo 使用权重自带的八步采样日程。

### SenseNova

SenseNova 使用官方原生推理包，需额外安装。已有官方推理环境时，也可以直接使用该环境生成，评分仍在 benchmark 环境执行。

```bash
git clone https://github.com/OpenSenseNova/SenseNova-U1.git
pip install --no-deps -e SenseNova-U1
python scripts/bench.py download --models sensenova-u1.5-8b-mot
python scripts/bench.py generate --models sensenova-u1.5-8b-mot --all --output-root outputs/run1
python scripts/bench.py evaluate --runs-root outputs/run1 --models sensenova-u1.5-8b-mot --workers 2 --device cpu
```

原生接口为 `it2i_generate`，使用官方 50 步、文本 CFG 4、图像 CFG 1、timestep shift 3、seed 42 和 SDPA。只输入当前原图及原始指令，不启用可选 Edit PE 或 think mode。输出尺寸显式设为原图补边后的尺寸；输入图保留原始像素，模型内部仍执行官方视觉编码预处理。输出张量按官方示例转换为 RGB PNG 并完整保留。当前适配需要模型完整驻留 GPU，不支持 `--cpu-offload`。

### Fibo Edit Turbo 和 JoyAI Plus

这两个模型使用 README 安装的同一套环境，无需另外创建环境或降级 Transformers：

```bash
python scripts/bench.py download --models fibo2-edit-turbo,joyai-image-edit-plus
python scripts/bench.py generate --models fibo2-edit-turbo,joyai-image-edit-plus --all --output-root outputs/run1
python scripts/bench.py evaluate --runs-root outputs/run1 --models fibo2-edit-turbo,joyai-image-edit-plus --workers 2 --device cpu
```

Fibo Edit Turbo 使用官方四步、CFG 1 的直接指令编辑接口，不启用可选的 VGL/Gemini 提示词转换，也不输入 mask。JoyAI Plus 使用官方 30 步、CFG 4，`images` 列表只放当前 case 的一张原图；保留官方内部参考图预处理，显式指定原图补边后的输出尺寸。两者均保存完整原始输出，评委仍读取原始编辑指令。

Fibo 权重需要先在 Hugging Face 模型页面接受访问条件。

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

默认沿用 TextEditBench 的 `temperature=0`、`max_tokens=500`，GPT-6-Luna 也使用这组参数。四个维度分别独立请求。

每个评测进程默认并发处理 5 个 case，每个 case 内的维度依次请求，同时进行的 API 请求不超过 5 个。用 `--workers 10` 改为 10 并发，`--workers 1` 使用串行。`bench.py evaluate` 和 `mllm_evaluate.py` 都支持此参数。

终端进度条显示已处理维度数、case 数、失败数、缓存命中数、API 请求次数、用时和预计剩余时间；`--no-progress` 可关闭显示。成功响应逐条保存，改变并发数不会使已匹配的缓存失效。中断后会保存已返回的请求结果，再释放输出锁。

只运行 MLLM 评分时，例如：

```bash
python scripts/mllm_evaluate.py \
  --generation-manifest outputs/run1/flux3-klein/generation_manifest.json \
  --output outputs/run1/flux3-klein/mllm_scores.json \
  --workers 10 --require-scoring
```

如需配置 API 地址、凭据来源、超时或重试次数，可以创建自己的 JSON，通过 `evaluate --config path/to/judge.json` 传入。字段示例见 [judge.example.json](../config/judge.example.json)，其中 `request_profile=texteditbench` 固定使用上述参数。

`reasoning_chat` 保留为显式可选的兼容配置，只有在 JSON 中指定时才启用。比较模型时应使用同一个评委和请求参数。

也可以将 `OPENAI_API_KEY`、`OPENAI_API_URL`、`OPENAI_MODEL` 写入仓库根目录的 `.env`，每行 `变量名=值`。脚本自动读取，已导出的环境变量优先。不要提交自己的密钥。

## 路径与输出尺寸

默认数据在 `data/EditableDesignBench/`，权重在 `models/`；可用 `EDITABLEDESIGNBENCH_DATASET_ROOT`、`EDITABLEDESIGNBENCH_MODEL_ROOT` 环境变量修改。

生成器只读取原图和指令，不读取目标图或 mask。原图只在右侧和底部补边到 32 的倍数。模型原始返回图完整保留，评测时在内存中去掉额外补边。直接提交的预测图须为原生尺寸 RGB PNG，不能将带补边图片当作原生尺寸图片提交。

RGBA 输出接受任意 Alpha 值，Alpha 仅用于诊断记录。评测输入直接取原始 RGB 三通道，不叠白底、不进行 Alpha 合成；原始 RGBA 文件保持不变。尺寸、补边、输入来源及 RGB 一致性检查仍会执行。

## 子集、续跑与汇总

首次测试可使用 `generate --ids 000_delete`。生成后评分会自动使用记录的同一组任务。为其他模型图片评分时，用 `evaluate --predictions predictions/MyModel --ids 000_delete` 指定子集，默认评测全部 400 例。

同一实验可以续跑，成功图片和匹配的评分会复用；修改任务集合、权重或生成参数后使用新的输出目录。

评分默认启用断点续跑，无需额外参数。重新执行同一条评分命令即可：MSE、PSNR、SSIM、LPIPS 分别记录成功状态；例如只有 LPIPS 失败时，仅补算 LPIPS。MLLM 同样只重试失败或未完成的 IF、TA、VC、LP，成功分数保留。`--no-resume` 才会强制全部重算，两种评分入口都支持。

复用前会重新校验原图、预测图、mask、评分配置和代码版本，因此仍会有文件读取；校验通过的成功项不会再次计算或调用 API。终端用 `cached`、`reused_metrics` 标明跳过项，报告的 `resume` 字段记录实际复用数量。兼容上一版的像素报告，会自动导入已有成功分数。

同一输出文件的 `.cache.json` 保存历史成功项，`.bak` 保留开始本轮前的报告。临时缺失的图片、子集测试或一次失败不会清掉历史成功项。改变图片、mask、评委或评分协议后，相应旧分数会失效；每个模型使用自己的结果与缓存目录。

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
