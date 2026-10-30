# Third-party attribution

EditableDesignBench evaluation and generation adapters are distributed under MIT. This license does not replace the licenses of the dataset, model checkpoints, or bundled third-party components.

- The dataset at https://huggingface.co/datasets/yomi-chen/EditableDesignBench is licensed under CC BY-NC 4.0.
- The four files in `vendor/texteditbench/evaluation/` are unmodified TextEditBench evaluation sources. Their original headers, upstream attribution, revision and per-file license declarations are retained. See `vendor/texteditbench/NOTICE.md` and `vendor/texteditbench/SOURCE_MANIFEST.json`.
- The bundled Diffusers wheel contains unmodified upstream Python sources at revision `d8c1b13c756df0059bcdb6e477a1eafcd502098d` from https://github.com/huggingface/diffusers. Diffusers is distributed under Apache 2.0; its license is included in the wheel. See `vendor/diffusers-source.json`.
- Model checkpoints are downloaded from their respective publishers and remain subject to their own terms. They are not included in this repository.

Research attribution: Rui Gui, Yang Wan, Haochen Han, Dongxing Mao, Fangming Liu, Min Li, and Alex Jinpeng Wang, *TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering*, arXiv:2512.16270, https://github.com/CSU-JPG/TextEditBench.
