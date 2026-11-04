# 进阶用法

常规安装、生成和评分见 [README](../README.md)。以下内容按需使用。

## 环境与依赖

推荐 Linux、Python 3.12，这是本项目已验证的 Python 版本。`environment.yml` 只创建同名 Python / pip 环境，与 README 的 `conda create` 等价，不会安装项目依赖。

- `requirements.txt`：下载数据、评测已有图片。包含像素指标需要的 PyTorch / TorchVision / TorchMetrics，以及 MLLM 的 HTTP、图片和进度工具；无需安装 OpenAI SDK。
- `requirements-generation.txt`：通过 `-r requirements.txt` 包含基础依赖，额外安装 Diffusers、Transformers、Accelerate 和 tokenizer 依赖。
- `requirements.lock.txt`：已验证的完整环境快照，包含直接与间接依赖，限定 Linux x86_64、Python 3.12、PyTorch 2.8.0 / CUDA 12.8。需要相同版本时，用它替代前两份清单安装；它不是 constraints 文件。

常规依赖采用版本范围，避免强制覆盖满足要求的环境。范围表示允许解析的版本，尚未逐一验证其中所有组合；精确复现实验请使用锁定文件。生成管线所需的 Diffusers 构建保留固定版本，原因和来源见 [NOTICE](../NOTICE.md)。

### CPU 与 CUDA

基础依赖不强制安装 CUDA 12.8 构建。希望只装 CPU 版时，在安装基础依赖前运行：

```bash
python -m pip install "torch>=2.8,<3" "torchvision>=0.23,<1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

本地生成需要 CUDA 版 PyTorch。例如安装 CUDA 12.8 构建：

```bash
python -m pip install "torch>=2.8,<3" "torchvision>=0.23,<1" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-generation.txt
```

其他 CUDA 版本按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)选择与驱动、GPU 匹配的构建，随后安装生成依赖。可用以下命令检查依赖一致性及当前进程的 GPU 可见性：

```bash
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
```

生成环境还可运行 `python scripts/bench.py check-env --prepare-metrics`，检查默认模型管线并预下载 LPIPS 感知网络权重；该命令需要完整生成依赖。LPIPS 在 CPU 上也可计算。

还原锁定环境时，在新的 Python 3.12 环境和仓库根目录中运行：

```bash
python -m pip install -r requirements.lock.txt
```

### 显存与模型依赖

本地生成在 H200 和 96 GB RTX PRO 6000 Blackwell 上验证过。不同模型、原图尺寸和 offload 设置的占用不同，尚无统一的最低显存承诺。先以 `--sample 1` 试跑；支持 offload 的模型可加 `--cpu-offload`。外部图片评分使用 `--device cpu` 即可，无需本地图像模型或显卡。

## 模型选择

替换 README 命令中的模型 ID 即可；多个 ID 用逗号分隔。以下模型共用生成依赖，SenseNova 另需安装原生包。

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

这两个模型使用 `requirements-generation.txt` 的同一套生成环境，无需另外创建环境或降级 Transformers：

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

## 文字与资产轨

文字任务来自 200 个原始设计，资产任务来自其中 34 个设计的 40 个资产单元。

下载器同时安装 400 条文字任务（数据根目录）和 200 条资产任务（`assets/`）。生成或评分外部图片时，用 `--track asset` 选择资产；`--dataset-root` 仍支持直接指定任意一条轨道的目录。生成实验会记录轨道与任务名单，后续评分自动读取。两轨使用不同提示词、掩码约定和缓存身份，分别汇报分数。

资产评委每次只接收两张原生尺寸整图（原图、模型结果），以及原始指令和操作类型。不裁取或放大局部，不读取 mask、参考目标或生成素材。mask 继续用于像素指标。

资产 MLLM 协议为 `editabledesignbench-asset-mllm-v3`。旧版四图和早期整图协议的分数与新版分开缓存；重跑原命令会重新计算资产 MLLM 分数，并保留历史缓存。文字 MLLM 协议和像素指标未改变。

| 资产维度 | 判断内容 |
|---|---|
| IF | 指定操作、目标、移动方向、放大/缩小是否符合指令 |
| OA | 资产状态是否正确完整：类别、数量、属性、必要的身份保持；删除时检查完整消失与残影 |
| VC | 编辑区域的画质、边缘、光照、阴影与风格融合；删除时检查背景修复 |
| LP | 其他文字、图片、图形及布局保持，避免新的遮挡和重叠 |

“稍向左”“变小”不要求精确等于参考位移或比例。只有指令要求同一人物时才检查身份；插入时不能要求模型恢复不可见的原人物。指令包含相框、圆环时整组作为资产单元。其他文字即使与新图片属性不一致，也不能擅自修改。允许区域内的非目标文字和图形同样受 LP 保护。

### 证据与错误判定

每次请求仍只评一个维度，独立核对原图中指定的对象/空位与结果中同一处的变化。新增图片不能由替换另一张已有图片代替；移动整张照片不能由移动照片内的主体代替；删除照片必须删除指定图片单元。

回复除分数和 `rationale` 外，还必须包含 `evidence`：`source`（原图状态）、`candidate`（结果状态）、`change`（实际变化）、`target_match`（`yes/no/uncertain`）、`operation`（`done/partial/not_done/wrong/uncertain`）、`defects`（可见缺陷列表）。证据应描述实际图像，而非复述指令。`target_match` 只判断是否定位并编辑了指定原图对象/空位，不能代替完成度：未编辑图可为 `target_match=yes, operation=not_done`；插入空位仍为空，也不意味着找错了空位。

- 明确改错对象时，IF / OA 最高 1 分；操作或方向错误时 IF 最高 1 分。身份保持完好可使 OA 保持高分，不能用 IF 的失败自动降低其他维度。
- 仅移动固定照片内部的主体，IF 最高 2 分；无法确认目标或操作完成时，不能给 IF 高分。
- 明确改变无关文字、其他图片或造成遮挡，LP 不能给 4–5 分；多处无关内容改变或严重遮挡最高 2 分。
- 正确、干净的结果仍可得 5 分。删除后必要的空白、合理的位置/尺度差异不作为缺陷。

缺少证据、格式错误或分数与同一回复中的事实声明冲突时，按无效响应重试；耗尽重试后保留错误和空分数，不自动截断、换算或补零。不同维度对目标/操作的事实判断矛盾时，结果中的 `consistency_flags` 与汇总的 `consistency_review` 标记待人工复核，原始分数不调整。跨维度存在不同分数本身不构成矛盾。

资产汇总提供四维均分、五种操作分项，以及四维均达到 4 分的联合合格率。未编辑源图可能获得较高 LP，但 IF 低、联合判定不合格，因此不使用四维简单平均作为唯一总分。缺图/坏图记录为模型输出失败，在全部所选任务的联合合格率分母中计失败；四维均分基于有效输出并报告覆盖率。API 错误记为待评，补齐之前正式均分和联合合格率保持空值。

像素公式仍复用 TextEditBench。资产轨将背景排除区域设为 `references.jsonl` 的 `pixel_preservation_mask`（`allowed_edit_mask.png`）。资产语义提示词与联合合格率属于本项目扩展协议，文字与资产分数分别报告。划分数据或估计置信区间时按 `source_id` 分组，同一设计的衍生任务保持在同组。

## API 与请求参数

默认沿用 TextEditBench 的 `temperature=0`、`max_tokens=500`。四个维度分别独立请求。完整文字集为 1,600 次请求，完整资产集为 800 次，不含重试。

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

首次测试可使用 `generate --sample 20`，默认种子为 42，按操作分层并优先选择不同原始设计；也可用 `--ids 000_delete` 指定任务。生成后评分自动使用记录的同一组任务。为其他模型图片评分时，用 `evaluate --predictions predictions/MyModel --ids 000_delete` 指定子集；资产图片添加 `--track asset`。默认评测整条轨道：文字 400 例、资产 200 例。

同一实验可以续跑，成功图片和匹配的评分会复用；修改任务集合、权重或生成参数后使用新的输出目录。

评分默认启用断点续跑，无需额外参数。重新执行同一条评分命令即可：MSE、PSNR、SSIM、LPIPS 分别记录成功状态；例如只有 LPIPS 失败时，仅补算 LPIPS。MLLM 同样只重试失败或未完成的维度：文字为 IF、TA、VC、LP，资产为 IF、OA、VC、LP，成功分数保留。`--no-resume` 才会强制全部重算，两种评分入口都支持。

复用前会重新校验对应输入、评分配置和代码版本；像素指标还校验 mask，资产整图 MLLM 不读取 mask。因此仍会有文件读取，校验通过的成功项不会再次计算或调用 API。终端用 `cached`、`reused_metrics` 标明跳过项，报告的 `resume` 字段记录实际复用数量。兼容上一版的像素报告，会自动导入已有成功分数。

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
