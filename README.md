<div align="center">

# 🎙️ ACRE

### An **A**coustic-**C**ue and Contextual-Knowledge **R**easoning **E**nhancement Framework for Speech Relation Extraction

<p>
  <a href="https://huggingface.co/datasets/DMU-ITREC/CSRTE-Speech"><img src="https://img.shields.io/badge/🤗%20Dataset-CSRTE--Speech-ffd21e" alt="Dataset"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Backbone-Qwen2--Audio--7B-6f42c1" alt="Backbone">
  <img src="https://img.shields.io/badge/Framework-MS--Swift-1f6feb" alt="MS-Swift">
  <img src="https://img.shields.io/badge/Training-LoRA%20SFT%20%2B%20GRPO-2ea44f" alt="Training">
</p>

<p>
  <a href="#-highlights">Highlights</a> •
  <a href="#-framework">Framework</a> •
  <a href="#-csrte-benchmark">Benchmark</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-main-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

**ACRE** extracts relational triples **directly from speech**. Instead of treating speech as "text in another modality", ACRE teaches a speech LLM to reason over both **_what is said_** (semantic content) and **_how it is said_** (pronunciation, tone and pauses). The reasoning is written out as a structured acoustic-semantic chain of thought before the model gives its final triples.

<p align="center">
  <img src="assets/motivation.png" width="95%" alt="Motivation">
</p>
<p align="center"><sub><b>Left:</b> acoustic cues such as pronunciation and pauses resolve entity ambiguity. <b>Right:</b> one pronunciation can map to several written forms, which biases character-level evaluation.</sub></p>

## ✨ Highlights

- 📚 **CSRTE**: a Chinese speech relation extraction benchmark recorded by real human speakers, covering **49 relation types** and **~20K utterances**.
- 🧠 **SAS-CoT**: a *Structured Acoustic-Semantic Chain-of-Thought* built from three reasoning chains:
  - **ARPC** (Accent-Robust Phonetic Chain): pronunciation and tone cues for resolving phonetic ambiguity.
  - **PGBC** (Prosody-Guided Boundary Chain): pause cues for locating entity boundaries.
  - **CGSC** (Context-Grounded Semantic Chain): contextual knowledge for entity disambiguation and relation reasoning.
- 🏋️ **Two-stage training**: LoRA SFT as a cold start, then **GRPO** with a composite reward that scores format, triplets, entities, relations, pronunciation and pauses.
- 📏 **PAE** (Pronunciation-Aware Evaluation): matches entities by tone-aware pinyin instead of exact characters, which removes homophone bias in Chinese SpeechRE evaluation.

## 🧩 Framework

<p align="center">
  <img src="assets/framework.png" width="100%" alt="ACRE framework">
</p>

| Stage | What happens | Entry point |
| :--- | :--- | :--- |
| **① SAS-CoT data construction** | Build reasoning-aware targets (ARPC + PGBC + CGSC → triples) | prepared JSONL data |
| **② Acoustic-semantic cold start** | LoRA SFT on Qwen2-Audio-7B to learn *speech → SAS-CoT → triples* | `scripts/train_sft.sh` |
| **③ Composite-reward RL** | GRPO from the SFT adapter, with ACRE rewards | `scripts/train_grpo.sh` + `acre/reward.py` |

### Output format

The model produces a reasoning block followed by its answer:

```text
<think>{"pronunciation": "nán jīng shì cháng jiāng dà qiáo", "pause": "南京市 <pause> 长江大桥"}</think>
<answer>长江大桥##位于##南京市</answer>
```

Multiple triples are written as `head##relation##tail` and joined with `$$`:

```text
head1##relation1##tail1$$head2##relation2##tail2
```

### Composite reward

`acre/reward.py` registers the MS-Swift reward plugin **`speechre_acre_reward`**:

| Component | Weight | Description |
| :--- | :---: | :--- |
| Triplet F1 | **0.80** | Exact match of the full `(head, relation, tail)` triple |
| Entity F1 | 0.05 | Correctness of head and tail entities |
| Relation F1 | 0.05 | Correctness of relation types |
| Format | 0.05 | Well-formed `<think>` JSON and `<answer>` block |
| Pronunciation | 0.025 | Similarity between the predicted and gold phonetic chain |
| Pause | 0.025 | Agreement between the predicted and gold pause boundaries |

Additional safeguards: malformed or overly long outputs receive a penalty (`-0.1`), and the reward is capped for repetitive generations, for answers with zero correct triples on non-empty samples, and for low-quality reasoning chains.

## 📚 CSRTE Benchmark

| Dataset | Speech | #Rel. | Train | Dev | Test | Avg. length |
| :--- | :---: | :---: | ---: | ---: | ---: | :---: |
| CoNLL04-SpeechRE | 🤖 TTS | 5 | 922 | 231 | 288 | 11.3s |
| ReTACRED-SpeechRE | 🤖 TTS | 40 | 33,477 | 9,350 | 5,805 | 12.9s |
| CommonVoice-SpeechRE | 🧑 Human | 45 | 14,557 | 2,495 | 2,494 | 11.6s |
| **CSRTE (Ours)** | 🧑 **Human** | **49** | **15,000** | **2,000** | **2,855** | **6.13s** |

📥 The data is hosted on Hugging Face: **[DMU-ITREC/CSRTE-Speech](https://huggingface.co/datasets/DMU-ITREC/CSRTE-Speech)**. It includes the speech recordings, transcriptions, named-entity annotations and relation annotations.

## 📁 Repository Structure

```text
ACRE/
├── acre/
│   ├── reward.py                    # ACRE composite reward (GRPO plugin)
│   ├── evaluate.py                  # Strict + Pronunciation-Aware Evaluation (PAE)
│   └── __init__.py
├── configs/
│   ├── paths.env.example            # Local path configuration template
│   └── README.md
├── patches/
│   ├── ms-swift-qwen2-audio.patch   # Qwen2-Audio compatibility patch for MS-Swift
│   └── README.md
├── requirements/
│   ├── environment.txt
│   └── ms-swift-version.txt
├── scripts/
│   ├── train_sft.sh                 # Stage ②: LoRA SFT
│   ├── train_grpo.sh                # Stage ③: GRPO
│   ├── evaluate.sh                  # Inference + metrics
│   └── README.md
├── assets/                          # Figures
└── README.md
```

## 🚀 Quick Start

### 1. Environment

```bash
conda create -n acre python=3.11 -y
conda activate acre
pip install -r requirements/environment.txt
```

ACRE runs on **[MS-Swift](https://github.com/modelscope/ms-swift)** `4.2.0.dev0` (the exact revision is recorded in `requirements/ms-swift-version.txt`). Install MS-Swift from source and apply the Qwen2-Audio compatibility patch:

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift && pip install -e . && cd ..
git -C ms-swift apply "$(pwd)/patches/ms-swift-qwen2-audio.patch"
```

> [!NOTE]
> This repository does not redistribute MS-Swift. Apply the patch to the revision listed in `requirements/ms-swift-version.txt`.

### 2. Configure paths

```bash
cp configs/paths.env.example configs/paths.env
vim configs/paths.env          # set BASE_MODEL / DATA_ROOT / OUTPUT_ROOT ...
source configs/paths.env
```

| Variable | Meaning |
| :--- | :--- |
| `BASE_MODEL` | Local path or hub ID of Qwen2-Audio-7B |
| `DATA_ROOT` | Directory with `train.jsonl`, `dev.jsonl`, `test.jsonl` and the audio files |
| `OUTPUT_ROOT` | Output directory for SFT, GRPO and evaluation results |
| `SFT_ADAPTER` | SFT LoRA checkpoint used to initialize GRPO |
| `GRPO_DATASET` | GRPO JSONL with a top-level `solution` field |
| `CUDA_VISIBLE_DEVICES` | GPUs to use |

### 3. Stage ②: LoRA SFT (cold start)

```bash
bash scripts/train_sft.sh
```

This reads `${DATA_ROOT}/train.jsonl` and `${DATA_ROOT}/dev.jsonl` and saves the adapter to `${OUTPUT_ROOT}/sft`.

<details>
<summary><b>Default SFT hyper-parameters</b></summary>

| Param | Value |
| :--- | :--- |
| LoRA rank / alpha | 8 / 32 (`all-linear`) |
| Learning rate | 2e-4 |
| Epochs | 10 |
| Batch size × grad. accum. | 2 × 16 |
| Max length | 2048 |
| Precision | bf16 |

</details>

### 4. Stage ③: GRPO with the ACRE reward

```bash
export SFT_ADAPTER=/path/to/sft/checkpoint
export GRPO_DATASET=/path/to/train_grpo.jsonl
bash scripts/train_grpo.sh
```

<details>
<summary><b>Default GRPO hyper-parameters</b></summary>

| Param | Value |
| :--- | :--- |
| Reward | `speechre_acre_reward` |
| Generations per prompt | 4 |
| Learning rate | 5e-6 |
| KL β | 0.04 |
| Epochs | 5 |
| Batch size × grad. accum. | 4 × 16 |
| Sampling | temperature 1.0, top-p 0.95 |
| Rollout engine | vLLM (colocate, 0.6 GPU mem.) |

</details>

## 📏 Evaluation

```bash
bash scripts/evaluate.sh /path/to/checkpoint test
```

The script runs greedy decoding with `swift infer` and then calls `acre/evaluate.py`. Results are written to `${OUTPUT_ROOT}/eval/<split>_<ckpt>.metrics.json` and report two settings:

- **Strict**: character-level exact match. Precision, recall and F1 for **triplets**, **entities** and **relations**.
- **PAE**: entities are mapped to tone-aware pinyin (e.g. `张莉 → zhang1 li4`) before matching, so homophone-equivalent predictions are not penalized as errors.

You can also score an existing prediction file directly:

```bash
python acre/evaluate.py --result_path preds.jsonl --label_path test.jsonl
```

## 📊 Main Results

End-to-end SpeechRE under strict evaluation (F1, %). Ent. = entity, Rel. = relation, Tri. = triplet.

| Method | CoNLL04<br>Ent. / Rel. / Tri. | ReTACRED<br>Ent. / Rel. / Tri. | CommonVoice<br>Ent. / Rel. / Tri. | **CSRTE**<br>Ent. / Rel. / Tri. |
| :--- | :---: | :---: | :---: | :---: |
| GPT-4o-Audio (ICL) | 31.21 / 59.57 / 5.64 | 13.21 / 41.61 / 1.14 | 29.33 / 31.70 / 3.12 | 35.12 / 40.26 / 8.72 |
| Qwen2-Audio (SFT) | 48.12 / 81.85 / 34.73 | 38.54 / 64.08 / 11.62 | 46.80 / 60.19 / 24.69 | 52.17 / 76.44 / 29.95 |
| LNA-ED | 19.13 / 56.32 / 11.12 | 18.26 / 43.15 / 3.67 | 27.61 / 38.51 / 6.01 | 40.32 / 50.15 / 17.21 |
| MCAM | 40.66 / 77.61 / 22.71 | 35.61 / 59.13 / 8.21 | 45.34 / 50.34 / 15.71 | 46.13 / 62.13 / 21.52 |
| RPG-MoGe | 50.21 / 79.64 / 24.67 | 36.76 / 58.38 / 9.18 | 47.20 / 53.48 / 18.29 | 52.70 / 69.48 / 26.33 |
| ACRE (Qwen2.5-Omni-3B) | 56.28 / 82.90 / 37.41 | 37.91 / 62.88 / 12.96 | 49.52 / 63.18 / 27.12 | 51.89 / 74.36 / 31.68 |
| ACRE (Qwen2.5-Omni-7B) | 57.84 / 85.73 / 40.31 | 40.83 / **72.09** / 17.64 | **54.35** / 67.41 / 29.02 | 54.83 / 78.21 / 33.98 |
| **ACRE (Qwen2-Audio-7B)** | **58.09** / **87.14** / **42.07** | **44.72** / 71.24 / **19.96** | 53.66 / **69.58** / **30.41** | **54.90** / **79.63** / **35.75** |

## 📝 Citation

If you find ACRE or CSRTE useful, please cite:

```bibtex
@article{ning2026acre,
  title  = {ACRE: An Acoustic-Cue and Contextual-Knowledge Reasoning Enhancement Framework for Speech Relation Extraction},
  author = {Ning, Jinzhong and Mu, Wenxuan and Zhao, Di and Zhang, Yijia and Lin, Hongfei},
  year   = {2026}
}
```

## 🙏 Acknowledgements

This project builds on [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio), [MS-Swift](https://github.com/modelscope/ms-swift), [TRL](https://github.com/huggingface/trl) and [vLLM](https://github.com/vllm-project/vllm). We thank the authors for their open-source contributions.

<div align="center">
<sub>⭐ If this repository helps your research, please give it a star!</sub>
</div>
