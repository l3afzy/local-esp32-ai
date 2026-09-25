# local-esp32-ai

A language model that runs entirely on an ESP32. No Wi-Fi, no cloud, no API.
It answers directly: the answer, nothing else.

```
you: how much does a hummingbird weigh
esp: Between 2 and 20 grams.
you: what is the capital of australia?
esp: Canberra.
you: what is 12 times 7
esp: 84
you: what is the capital of bolivia
esp: I don't know.
```

Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum and shrinking every piece down to microcontroller size:

| Curriculum lesson | Used here |
|---|---|
| [07-07 GPT causal language modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) | `train/model.py`, a 4-layer decoder-only transformer |
| [10-01 Tokenizers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/01-tokenizers) | character tokenizer, 97 tokens, no vocab file needed |
| [10-06 Instruction tuning (SFT)](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/06-instruction-tuning-sft) | loss on answer tokens only, so the model learns to answer and stop |
| [10-11 Quantization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/11-quantization) | `train/export.py`, int8 weights with per-row scales |
| [07-12 KV cache](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) / [10-12 Inference optimization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/12-inference-optimization) | `firmware/lib/tinyai/tinyai.c`, int8 matmuls and an int8 KV cache |

## How it stays direct

Directness is enforced at every layer, not just requested in a prompt:

1. **Data.** Every training answer in `data/facts.tsv` is the answer only: no preamble, no restated question.
2. **Loss.** The loss covers answer characters only (`train/train.py`). The model never learns to write anything before the answer.
3. **Stop token.** Each answer ends in EOS, and generation stops there.
4. **Greedy decoding.** It always takes the most likely token: no sampling, no temperature, no rambling.
5. **Hard cap.** An answer is at most 48 characters, enforced in C.
6. **Knowledge gate** (`gate.c`). A 700k-parameter model can't tell what it doesn't know, so it would invent a capital for Bolivia. Before generating, the gate checks the question's content words (typos, plurals and filler words tolerated) against the questions the model trained on. With no match, the answer is `I don't know.`
7. **Calculator** (`calc.c`). Arithmetic goes to a parser, not the model: `12*7`, `15% of 80`, `2^10`, `100 divided by 8`.

## Quick start: flash the prebuilt model

You need [PlatformIO](https://platformio.org/install/cli) and any ESP32 board.

```sh
cd firmware
pio run -t upload            # classic ESP32 (esp32dev)
pio run -e esp32-s3 -t upload
pio device monitor           # type questions, press Enter
```

The trained model is already in `firmware/src/model_data.h` (flash) and `firmware/data/model.bin` (PC).

## Try it on a PC first

The PC build compiles the exact same C engine as the firmware:

```sh
make -C host
./host/tinyai firmware/data/model.bin
```

## Train your own

```sh
pip install -r requirements.txt
python train/train.py          # about 25 min on a 4-core CPU, no GPU needed
python train/export.py         # int8 -> firmware/data/model.bin + firmware/src/model_data.h
make -C host && python train/evaluate.py
```

To teach it new facts, add lines to `data/facts.tsv` and retrain:

```
question|paraphrase|another paraphrase<TAB>Direct answer.
```

Training adds noise to the questions: typos, "hey", "tell me", "please", contractions. You only need to write the clean questions.

## Specs

| | |
|---|---|
| Architecture | GPT: 4 layers, dim 128, 4 heads, MLP 384, context 96, tied embeddings |
| Parameters | 681,216 |
| Model in flash | 720 KB (int8 + per-row fp32 scales + 639 gate questions) |
| RAM at runtime | about 115 KB (int8 KV cache 98 KB, activations and scales about 17 KB) |
| Engine code | about 8 KB of Xtensa code, pure C99, no dependencies |
| Works on | ESP32, ESP32-S3, ESP32-C3 (anything with 4 MB flash and 160 KB free heap) |

RESULTS_PLACEHOLDER

## Layout

```
data/facts.tsv              what the model knows (edit this)
data/eval.tsv               held-out questions: paraphrases, must-refuse, arithmetic
train/common.py             tokenizer, normalization, augmentation (mirrored in C)
train/model.py              the transformer
train/train.py              training
train/export.py             int8 quantization + export
train/evaluate.py           scores the real C engine
firmware/lib/tinyai/        inference engine, gate, calculator (C99)
firmware/src/main.cpp       serial chat for the ESP32
host/                       PC build of the same engine
```

## Limits

- It knows about 300 facts, the ones in `data/facts.tsv`. Anything else gets `I don't know.`, which is the point.
- 700k parameters can memorize facts and handle rephrasing. They can't reason. Multi-step questions are out of scope.
- The gate matches words, not meaning. Heavily reworded questions can be refused even when the fact is known. Add those phrasings to `facts.tsv`.
