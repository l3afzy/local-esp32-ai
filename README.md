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
you: 10 km in miles
esp: 6.21371 miles
you: melting point of gallium
esp: 29.8 C.
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
   │ normalize         lowercase, strip junk            "whats the capital of iceland"
   │ compute           arithmetic, units, bases, primes  not computable
   │ knowledge gate    indexed match, 7,796 phrasings    fact #1,204
   │                   no match -> "I don't know."
   │ model (int4 GPT)  fact's shortest phrasing -> answer "capital of iceland"
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
7. **Computed answers** (`calc.c`, `convert.c`). Anything that can be calculated is, rather than memorized: arithmetic (`12*7`, `15% of 80`, `square root of 144`), unit conversion across 55 units (`10 km in miles`, `how many feet in a mile`, `100 f in celsius`), number bases (`255 in hex`), Roman numerals (`1994 in roman numerals`) and primes (`is 91 prime` → `No, 91 = 7 x 13.`).

## Optimizations

Three rounds so far, each measured on the C engine:

| | v1 | v2 | **v3 (now)** |
|---|---|---|---|
| Facts the model stores | 309 | 1,089 | **2,973** |
| Accepted phrasings | 639 | 2,415 | **7,796** |
| Questions answered by computation | arithmetic | arithmetic | **+ units, bases, Roman numerals, primes** |
| Model file (flash) | 720 KB | 494 KB | **551 KB** |
| Weights | int8 | int4 + QAT | int4 + QAT |
| Gate time per question at 7,796 phrasings (x86) | n/a | 18 ms | **0.25 ms** |
| Forward passes per answer (same facts) | n/a | 46.5 | **36.1** |
| Matmuls on dual-core chips | 1 core | 1 core | **2 cores** |
| Trained phrasings correct | 97% | 100% | **100% (7,797/7,797)** |
| Held-out questions correct | 85% | 100% | **100% (124/124)** |

**v2: fit more facts per byte**
- **Retrieval-canonical prompts.** The gate finds the fact, and the model only ever sees that fact's canonical key, never the raw text. Capacity goes to facts instead of typos, and accuracy on the chip matches training.
- **int4 weights with quantization-aware training.** 4.5 bits per weight: one fp32 scale per 32 weights. The second half of training uses the rounded weights through a straight-through estimator, and `export.py` asserts its rounding matches training bit for bit.
- **Flash bandwidth is the real bottleneck.** Weights stream from memory-mapped flash through a 32 KB cache. The int4 kernel reads 8 weights per 32-bit load and uses unsigned nibbles with a precomputed-sum correction: Σ(w−8)·x = Σw·x − 8·Σx.
- **Grouped-query attention** halves the K/V projections and the KV cache.

**v3: push more knowledge through the same 616K-parameter model**
- **Bulk facts from real datasets, not memory.** Every chemical element (symbol, number, mass, melting and boiling points, density, group, period, type, state) comes from [mendeleev](https://github.com/lmmentel/mendeleev) (MIT). Currencies come from Unicode CLDR via Babel (BSD), calling codes from phonenumbers (Apache 2.0), and 20 CODATA physical constants from SciPy (BSD). Data that didn't pass checking is left out. mendeleev's discoverer list has misspellings and wrong entries, so discoverers are hand-curated for 32 well-documented elements, plus 10 known since antiquity. CLDR's official-language data is misleading (South Africa: English only), so languages aren't included.
- **Computation instead of memorization.** Unit conversion, number bases, Roman numerals and primes cover unlimited questions in about 9 KB of code, and 40 memorized facts became redundant and were removed. What is ambiguous stays with the facts: "ton" could be US or metric, and "ounces in a cup" mixes mass and volume.
- **An indexed gate.** Phrasings are stored as word IDs into a 1,610-word dictionary, with the question type precomputed. That's 97 KB instead of 312 KB of text. A question fuzzy-matches each of its words against the dictionary once, then scores 7,796 phrasings with integer compares: 18 ms → 0.25 ms on x86, about 70× faster. The index is built in Python at export time, and `evaluate.py` proves it parses all 7,796 phrasings exactly like the C gate.
- **Shortest phrasing as the model's key.** On the chip every prompt character is a full forward pass, so each fact's key is its shortest accepted phrasing ("gold melting point", not "what is the melting point of gold"). That's 22% fewer forward passes per answer, and it fixed a context overflow on the longest question.
- **Both cores.** On the ESP32 and ESP32-S3 each matmul is split across the two cores with FreeRTOS task notifications. Output rows are independent, so answers are bit-identical to single-core, which the emulator run confirms.

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
python train/train.py          # about 45 min on a 4-core CPU, no GPU needed
python train/export.py         # int4 + gate index -> firmware/data/model.bin + firmware/src/model_data.h
make -C host && python train/evaluate.py
```

To teach it new facts, add lines to `data/facts.tsv` and retrain. Every phrasing is accepted by the gate; the model learns the shortest one:

```
question|another phrasing|a third phrasing<TAB>Direct answer.
```

Templated and dataset-driven facts (capitals, elements, currencies, calling codes, constants, authors, ...) come from `data/more_facts.py`:

```sh
pip install mendeleev babel phonenumbers scipy
python data/more_facts.py > data/facts_generated.tsv
```

## Specs

| | |
|---|---|
| Architecture | GPT: 4 layers, dim 128, 4 query heads / 2 K/V heads, MLP 384, context 96, tied embeddings |
| Parameters | 615,680 |
| Knowledge | 2,973 facts, 7,796 accepted phrasings, plus computed answers |
| Model in flash | 551 KB: int4 weights 364 KB, int8 embeddings 25 KB, fact keys 64 KB, gate index 97 KB |
| RAM at runtime | about 65 KB (int8 KV cache 49 KB, activations, scales and gate buffers about 16 KB) |
| Engine code | about 21 KB of Xtensa code and tables, pure C99, no dependencies |
| Works on | ESP32, ESP32-S3, ESP32-C3 |

## Results

Scored by `train/evaluate.py`, which runs the real C engine (the same code the ESP32 runs), not the Python model:

| Test | Score |
|---|---|
| Every trained phrasing | 7,797 / 7,797 |
| Held-out rephrasings the model never saw | 82 / 82 |
| ...including traps (Iceland vs Ireland, Guinea vs Guinea-Bissau, Sudan vs South Sudan, Fahrenheit→Celsius vs Celsius→Fahrenheit) | all correct |
| Unknown topics that must be refused | 19 / 19 |
| Computed answers (arithmetic, units, bases, Roman numerals, primes) | 23 / 23 |
| Gate index parses like the C gate | 7,796 / 7,796 |
| Answers over 48 chars or starting with filler | 0 |

Speed on a laptop CPU: 12 ms per model answer, 0.25 ms to refuse, and effectively instant for computed answers.

### Verified on ESP32 firmware

- **Builds** with PlatformIO for all three targets with no warnings:

  | Board | Flash used (of 3 MB app partition) | Static RAM |
  |---|---|---|
  | ESP32 | 869 KB (28%) | 25.1 KB |
  | ESP32-S3 | 866 KB (28%) | 22.1 KB |
  | ESP32-C3 | 852 KB (27%) | 17.4 KB |

- **Runs** in Espressif's ESP32 emulator (QEMU) as the real flash image, on both cores: it boots with 275 KB of heap free and answers all 124 held-out questions plus a random 1,000 of the trained phrasings exactly as the PC build does.
- **Not yet timed on physical hardware.** Emulator timings aren't real. To measure on a board, set `SHOW_TIMING 1` in `firmware/src/main.cpp`.

Re-run the emulator check yourself (needs [Espressif's QEMU](https://github.com/espressif/qemu/releases)):

```sh
cd firmware && pio run -e esp32dev
echo "how much does a hummingbird weigh" | python qemu_test.py --qemu path/to/qemu-system-xtensa
```

## Layout

```
data/facts.tsv              hand-written facts (edit this)
data/more_facts.py          generator for templated and dataset facts -> data/facts_generated.tsv
data/eval.tsv               held-out questions: paraphrases, must-refuse, arithmetic
train/common.py             tokenizer and normalization (mirrored in C)
train/model.py              the transformer, int4 fake-quantization for QAT
train/train.py              training
train/export.py             int4 quantization + export
train/gate_index.py         builds the gate's word index (mirrors gate.c)
train/evaluate.py           scores the real C engine
firmware/lib/tinyai/        inference engine, gate, calculator, unit converter (C99)
firmware/src/main.cpp       serial chat for the ESP32
firmware/qemu_test.py       runs the built firmware in the ESP32 emulator
host/                       PC build of the same engine
```

## Limits

- It knows what's in `data/facts*.tsv` and nothing else. Anything else gets `I don't know.`, which is the point.
- The gate matches words, not meaning. Heavily reworded questions can be refused even when the fact is known. Add those phrasings to the fact's line.
- 616K parameters can store facts. They can't reason. Multi-step questions are out of scope.
- Numbers from datasets are as good as the datasets: element data is from mendeleev, currencies from CLDR as of Babel 2.18.
