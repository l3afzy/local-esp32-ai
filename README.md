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
you: whats the safe temp for chicken
esp: 165 F (74 C).
you: 15% tip on 42.50
esp: Tip 6.38, total 48.88.
you: how long from 9am to 5:30pm
esp: 8 hours 30 minutes
you: what day of the week was july 20 1969
esp: Sunday.
you: emergency number uk
esp: 999 or 112.
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
   │ compute           math, money, units, times, dates  not computable
   │ knowledge gate    indexed match, 8,521 phrasings    fact #1,204
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
7. **Computed answers** (`calc.c`, `convert.c`, `everyday.c`). Anything that can be calculated is, rather than memorized, and number words work too ("twelve times seven", "half of 30", "a dozen"):
   - math: `12*7`, `15% of 80`, `50 plus 10%` (= 55, like any calculator), `square root of 144`, `average of 3, 5 and 10`
   - money: `15% tip on 42.50` → `Tip 6.38, total 48.88.`, `20% off 80`, `what percent is 12 of 48`, `split 90 between 4 people`
   - 63 units: `10 km in miles`, `how many cups in a liter`, `350 f in c`, `32 psi in bar`
   - times: `3pm in 24 hour time`, `how long from 9am to 5:30pm`
   - dates: `what day was july 20 1969`, `days between march 3 and june 10`, `30 days after march 3 2025`, `is 2100 a leap year`
   - numbers: `255 in hex`, `1994 in roman numerals`, `is 91 prime` → `No, 91 = 7 x 13.`

## Optimizations

Four rounds so far, each measured on the C engine:

| | v1 | v2 | v3 | **v4 (now)** |
|---|---|---|---|---|
| Facts the model stores | 309 | 1,089 | 2,973 | **3,230** |
| Accepted phrasings | 639 | 2,415 | 7,796 | **8,521** |
| Computed questions | arithmetic | arithmetic | + units, bases, Roman numerals, primes | **+ money, tips, percentages, times, dates, number words** |
| Model file (flash) | 720 KB | 494 KB | 551 KB | **566 KB** |
| Weights | int8 | int4 + QAT | int4 + QAT | int4 + QAT |
| Gate time per question (x86) | n/a | 18 ms | 0.25 ms | **0.32 ms** |
| Matmuls on dual-core chips | 1 core | 1 core | 2 cores | 2 cores |
| Trained phrasings correct | 97% | 100% | 100% | **100% (8,522/8,522)** |
| Held-out questions correct | 85% | 100% | 100% (124) | **100% (214/214)** |
| Everyday questions, natural phrasing | n/a | n/a | 79% (71/90) | **100% (90/90)** |

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

**v4: smart and reliable for everyday use**
- **Everyday knowledge.** Safe cooking temperatures (USDA), cooking basics, emergency numbers for 18 countries and the EU, the US poison control and 988 crisis lines, standard first aid (Red Cross / NHS), health reference numbers (CDC / AHA), household safety, money, holidays, daylight saving, and device shortcuts.
- **Everyday math in code** (`everyday.c`): tips, discounts, percentages, bill splitting, averages, 12/24-hour time, durations across midnight, day of the week, days between dates, date arithmetic, leap years. Number words are understood. "50 plus 10%" means 55. Money rounds correctly (48.875 → 48.88, not the 48.87 floating point gives).
- **A gate that understands how people ask**, measured on 90 new everyday questions written the way people type them. The first run got 71 right, and only one of the 19 misses was a wrong answer, not an "I don't know". Fixes that generalize: filler words ("should", "need", "make"), contractions ("when's", "isn't"), synonyms (temp→temperature, stay→last, replace→change), and treating "what date" and "when" as the same question. Two relaxations made it *worse* and were reverted. Letting any "how X" question match a "what" question turned "how far is pluto" into "what is pluto". Ignoring the word after "how" made the same question match "is pluto a planet". The final gate refuses all 30 must-refuse questions.

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
| Knowledge | 3,230 facts, 8,521 accepted phrasings, plus computed answers |
| Model in flash | 566 KB: int4 weights 364 KB, int8 embeddings 25 KB, fact keys 70 KB, gate index 107 KB |
| RAM at runtime | about 65 KB (int8 KV cache 49 KB, activations, scales and gate buffers about 16 KB) |
| Engine code | about 36 KB of Xtensa code and tables, pure C99, no dependencies |
| Works on | ESP32, ESP32-S3, ESP32-C3 |

## Results

Scored by `train/evaluate.py`, which runs the real C engine (the same code the ESP32 runs), not the Python model:

| Test | Score |
|---|---|
| Every trained phrasing | 8,522 / 8,522 |
| Held-out rephrasings of general facts | 83 / 83 |
| ...including traps (Iceland vs Ireland, Guinea vs Guinea-Bissau, Sudan vs South Sudan, Fahrenheit→Celsius vs Celsius→Fahrenheit) | all correct |
| Everyday questions, phrased the way people type them (29 of them computed) | 80 / 80 |
| Unknown topics that must be refused ("what time is it in tokyo", "how many calories are in a big mac") | 30 / 30 |
| Calculator, units, bases, primes | 21 / 21 |
| Gate index parses like the C gate | 8,521 / 8,521 |
| Answers over 48 chars or starting with filler | 0 |

Speed on a laptop CPU: about 15 ms per model answer, 0.3 ms to refuse, and effectively instant for computed answers.

### Verified on ESP32 firmware

- **Builds** with PlatformIO for all three targets with no warnings:

  | Board | Flash used (of 3 MB app partition) | Static RAM |
  |---|---|---|
  | ESP32 | 920 KB (29%) | 25.1 KB |
  | ESP32-S3 | 916 KB (29%) | 22.1 KB |
  | ESP32-C3 | 901 KB (29%) | 17.4 KB |

- **Runs** in Espressif's ESP32 emulator (QEMU) as the real flash image, on both cores: it boots with 273 KB of heap free and answers all 214 eval questions plus a random 800 of the trained phrasings exactly as expected (1,014 / 1,014).
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
data/eval.tsv               held-out questions: paraphrases, everyday, must-refuse, computed
train/common.py             tokenizer and normalization (mirrored in C)
train/model.py              the transformer, int4 fake-quantization for QAT
train/train.py              training
train/export.py             int4 quantization + export
train/gate_index.py         builds the gate's word index (mirrors gate.c)
train/evaluate.py           scores the real C engine
firmware/lib/tinyai/        inference engine, gate, calculator, units, everyday math (C99)
firmware/src/main.cpp       serial chat for the ESP32
firmware/qemu_test.py       runs the built firmware in the ESP32 emulator
host/                       PC build of the same engine
```

## Limits

- It knows what's in `data/facts*.tsv` and nothing else. Anything else gets `I don't know.`, which is the point.
- No clock or internet: "what time is it", "weather", "news" and "how many days until Christmas" can't be answered.
- First-aid and health answers are short reference facts, not medical advice. In an emergency, call the local emergency number.
- The gate matches words, not meaning. Heavily reworded questions can be refused even when the fact is known. Add those phrasings to the fact's line.
- 616K parameters can store facts. They can't reason. Multi-step questions are out of scope.
- Numbers from datasets are as good as the datasets: element data is from mendeleev, currencies from CLDR as of Babel 2.18.
