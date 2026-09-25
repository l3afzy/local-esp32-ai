# local-esp32-ai

A language model that runs entirely on an ESP32. No Wi-Fi, no cloud, no API.
It answers directly: the answer, nothing else.

```
you: how much does a hummingbird weigh
esp: Between 2 and 20 grams.
you: capital of iceland?
esp: Reykjavik.
you: what is 12 times 7
esp: 84
you: how many moons does jupiter have
esp: I don't know.
```

Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum and shrinking every piece down to microcontroller size:

| Curriculum lesson | Used here |
|---|---|
| [07-07 GPT causal language modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) | `train/model.py`, a 4-layer decoder-only transformer |
| [07-15 Attention variants](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/15-attention-variants) | grouped-query attention: 4 query heads share 2 K/V heads |
| [10-01 Tokenizers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/01-tokenizers) | character tokenizer, 97 tokens, no vocab file needed |
| [10-06 Instruction tuning (SFT)](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/06-instruction-tuning-sft) | loss on answer tokens only, so the model learns to answer and stop |
| [10-11 Quantization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/11-quantization) | int4 weights, quantization-aware training |
| [07-12 KV cache](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) / [10-12 Inference optimization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/12-inference-optimization) | `tinyai.c`: integer matmuls, int8 KV cache |

## How a question is answered

```
"Whats the capital of Iceland??"
   │ normalize         lowercase, strip junk          "whats the capital of iceland"
   │ calculator        arithmetic? answer exactly      no
   │ knowledge gate    match against 2,415 phrasings  fact #412
   │                   no match -> "I don't know."
   │ model (int4 GPT)  canonical question -> answer    "what is the capital of iceland"
   ▼                   greedy, stops at EOS, 48 chars max
"Reykjavik."
```

## How it stays direct

Directness is enforced at every layer, not just requested in a prompt:

1. **Data.** Every training answer in `data/facts*.tsv` is the answer only: no preamble, no restated question.
2. **Loss.** The loss covers answer characters only. The model never learns to write anything before the answer.
3. **Stop token.** Each answer ends in EOS, and generation stops there.
4. **Greedy decoding.** It always takes the most likely token: no sampling, no temperature, no rambling.
5. **Hard cap.** An answer is at most 48 characters, enforced in C.
6. **Knowledge gate** (`gate.c`). A tiny model can't tell what it doesn't know. Ask for a capital it never saw and it will invent one. The gate checks the question against every phrasing the model was trained on. It tolerates typos, plurals, split words ("humming bird") and filler, and it requires the question type to agree: a "how many" question never gets a "what" answer. With no match, the answer is `I don't know.`
7. **Calculator** (`calc.c`). Arithmetic goes to a parser, not the model: `12*7`, `15% of 80`, `2^10`, `square root of 144`, `5 squared`.

## The optimizations that fit 1,089 facts in 494 KB

| | Before | After |
|---|---|---|
| Facts | 309 | **1,089** |
| Model in flash | 720 KB | **494 KB** |
| Weight format | int8 | **int4**, groups of 32, quantization-aware training |
| Attention | 4 K/V heads | **2 K/V heads** (GQA): half the KV cache |
| KV cache (RAM) | 98 KB | **49 KB** |
| Prompt the model sees | raw user text | **canonical question** from the gate |
| Trained questions correct | 622/640 (97%) | **2,416/2,416 (100%)** |
| Held-out questions correct | 58/68 (85%) | **100/100 (100%)** |

- **Retrieval-canonical prompts.** This matters most. The old model had to map every typo and rephrasing to an answer, which spent capacity on noise and still failed on paraphrases ("France capital" → "Beijing."). Now the gate finds the fact and the model only ever sees that fact's canonical question. Every parameter goes to storing facts, and accuracy on the chip matches accuracy in training.
- **int4 with quantization-aware training.** Weights are 4-bit, one fp32 scale per 32 weights: 4.5 bits per weight. For the second half of training the forward pass uses the rounded int4 weights (a straight-through estimator), so the network learns weights that survive rounding. `export.py` asserts that its rounding matches training bit for bit.
- **Flash bandwidth is the real bottleneck.** On an ESP32 the weights stream from memory-mapped flash through a 32 KB cache, so fewer bytes per weight means faster tokens, not just a smaller file. The int4 kernel reads 8 weights per 32-bit load. It uses unsigned nibbles and corrects with a precomputed sum, Σ(w−8)·x = Σw·x − 8·Σx, which saves a subtract per weight.
- **Grouped-query attention** halves the K/V projections and the KV cache.

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
./host/tinyai firmware/data/model.bin           # chat
./host/tinyai firmware/data/model.bin --why     # also show which fact the gate matched
```

## Train your own

```sh
pip install -r requirements.txt
python train/train.py          # about 16 min on a 4-core CPU, no GPU needed
python train/export.py         # int4 -> firmware/data/model.bin + firmware/src/model_data.h
make -C host && python train/evaluate.py
```

To teach it new facts, add lines to `data/facts.tsv` and retrain. The first question is the canonical one the model learns; the others are phrasings the gate accepts:

```
question|another phrasing|a third phrasing<TAB>Direct answer.
```

Templated facts (capitals, elements, authors, ...) come from `data/more_facts.py`:

```sh
python data/more_facts.py > data/facts_generated.tsv
```

## Specs

| | |
|---|---|
| Architecture | GPT: 4 layers, dim 128, 4 query heads / 2 K/V heads, MLP 384, context 96, tied embeddings |
| Parameters | 615,680 |
| Knowledge | 1,089 facts, 2,415 accepted phrasings |
| Model in flash | 494 KB (int4 weights, int8 embeddings, fact tables) |
| RAM at runtime | about 60 KB (int8 KV cache 49 KB, activations and scales about 10 KB) |
| Engine code | about 11 KB of Xtensa code, pure C99, no dependencies |
| Works on | ESP32, ESP32-S3, ESP32-C3 |

## Results

Scored by `train/evaluate.py`, which runs the real C engine (the same code the ESP32 runs), not the Python model:

| Test | Score |
|---|---|
| Every trained phrasing (2,416) | 2,416 / 2,416 |
| Held-out rephrasings the model never saw | 69 / 69 |
| Unknown topics that must be refused | 19 / 19 |
| ...including traps (Iceland vs Ireland, Guinea vs Guinea-Bissau, Sudan vs South Sudan) | all correct |
| Arithmetic | 12 / 12 |
| Answers over 48 chars or starting with filler | 0 |

### Verified on ESP32 firmware

- **Builds** with PlatformIO for all three targets:

  | Board | Flash used (of 3 MB app partition) | Static RAM |
  |---|---|---|
  | ESP32 | 793 KB (25%) | 21.9 KB |
  | ESP32-S3 | 789 KB (25%) | 18.8 KB |
  | ESP32-C3 | 775 KB (25%) | 14.2 KB |

- **Runs** in Espressif's ESP32 emulator (QEMU) as the real flash image: it boots, loads the model with 283 KB of heap to spare, and gives the same answers as the PC build on all 100 held-out questions and all 2,416 trained phrasings.
- **Not yet timed on physical hardware.** Emulator timings aren't real. A rough estimate from the instruction count is 1 to 2 seconds per answer on a classic ESP32 at 240 MHz; on a laptop it takes about 20 ms. To measure on a board, set `SHOW_TIMING 1` in `firmware/src/main.cpp`.

Re-run the emulator check yourself (needs [Espressif's QEMU](https://github.com/espressif/qemu/releases)):

```sh
cd firmware && pio run -e esp32dev
echo "how much does a hummingbird weigh" | python qemu_test.py --qemu path/to/qemu-system-xtensa
```

## Layout

```
data/facts.tsv              hand-written facts (edit this)
data/more_facts.py          generator for templated facts -> data/facts_generated.tsv
data/eval.tsv               held-out questions: paraphrases, must-refuse, arithmetic
train/common.py             tokenizer and normalization (mirrored in C)
train/model.py              the transformer, int4 fake-quantization for QAT
train/train.py              training
train/export.py             int4 quantization + export
train/evaluate.py           scores the real C engine
firmware/lib/tinyai/        inference engine, gate, calculator (C99)
firmware/src/main.cpp       serial chat for the ESP32
firmware/qemu_test.py       runs the built firmware in the ESP32 emulator
host/                       PC build of the same engine
```

## Limits

- It knows what's in `data/facts*.tsv` and nothing else. Anything else gets `I don't know.`, which is the point.
- The gate matches words, not meaning. Heavily reworded questions can be refused even when the fact is known. Add those phrasings to the fact's line.
- 600k parameters can store facts. They can't reason. Multi-step questions are out of scope.
