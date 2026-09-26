# local-esp32-ai: details

The [README](../README.md) is the short version. This is how it works and how it was tested.

## Where it comes from

Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum and shrinking every piece down to microcontroller size:

| Curriculum lesson | Used here |
|---|---|
| [07-07 GPT causal language modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) | `train/model.py`, a 6-layer decoder-only transformer |
| [07-15 Attention variants](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/15-attention-variants) | grouped-query attention: 4 query heads share 2 K/V heads |
| [10-01 Tokenizers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/01-tokenizers) | character tokenizer, 97 tokens, no vocab file needed |
| [10-06 Instruction tuning (SFT)](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/06-instruction-tuning-sft) | loss on answer tokens only, so the model learns to answer and stop |
| [10-11 Quantization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/11-quantization) | int4 weights with fp16 scales, quantization-aware training |
| [07-12 KV cache](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) / [10-12 Inference optimization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/12-inference-optimization) | `tinyai.c`: integer matmuls, int8 KV cache, both cores |

## Survival knowledge

197 facts across 15 topics, each answer following the source named in its section of `data/facts_survival.tsv`:

| Topic | Facts | Examples | Source |
|---|---|---|---|
| Priorities, being lost, kits | 18 | rule of threes, stay put, what to pack | US Army FM 21-76, Ready.gov |
| Water | 25 | boiling, bleach and iodine doses, SODIS, finding water, seawater, ORS recipe | CDC, EPA, WHO, FM 21-76 |
| Fire | 14 | tinder, fire without matches, wet wood, putting it out | US Forest Service, FM 21-76 |
| Shelter and warmth | 10 | debris hut, lean-to, snow cave, layers, cotton | FM 21-76, NWS |
| Signaling and rescue | 9 | 3 of anything, SOS, signal mirror, ground-to-air V and X | FM 21-76, ICAO, FCC |
| Navigation | 9 | North Star, Southern Cross, shadow stick, watch method, moss myth | FM 21-76 |
| Food | 10 | insects, mushrooms, berries, edibility test, milky sap | FM 21-76 |
| Cold | 20 | hypothermia, frostbite, falling through ice, ice thickness, 1-10-1 | CDC, MN DNR, US Coast Guard |
| Heat | 7 | heat exhaustion vs heat stroke, desert travel | CDC |
| Wildlife | 16 | black vs grizzly bears, mountain lions, moose, snakebite, ticks | National Park Service, CDC, WHO |
| Weather and disasters | 29 | lightning, tornado, flood, tsunami, wildfire, avalanche, power outage, generators | NWS, Ready.gov, FoodSafety.gov |
| Water hazards | 6 | rip currents, sinking car, river crossing, capsized boat | NOAA, US Coast Guard |
| First aid | 15 | bleeding, tourniquet, splints, shock, altitude sickness | Stop the Bleed, Red Cross, NHS, CDC |
| Knots | 6 | bowline, sheet bend, taut-line, clove, trucker's hitch | |
| Gear | 3 | survival knife, paracord | |

**How the answers were chosen.** Only guidance published by the agencies above, in their words where possible, cut to 48 characters. Where sources disagree, the more cautious answer wins. For example, a debris hut's leaf pile is given as 3+ feet (1 m), not the thinner 2 feet some guides say. Regional advice that doesn't generalize was left out: snakebite pressure bandages are Australian practice, and jellyfish advice depends on the species.

**Survival math is computed, not memorized:**

| You type | It answers | Rule |
|---|---|---|
| `how much water for 4 people for 3 days` | `12 gallons (45 L) for 3 days.` | Ready.gov: 1 gallon per person per day |
| `how much water should i keep for a family of 4` | `4 gallons (15 L) a day.` | |
| `thunder 10 seconds after lightning` | `About 2.1 miles (3.4 km). Go indoors.` | sound travels 343 m/s; NWS: if you hear thunder, go in |
| `how much bleach for 4 gallons` | `32 drops of 6% bleach; wait 30 min.` | CDC/EPA: 8 drops per gallon, 2 per liter |

**How the gate handles survival questions.** People in trouble describe their situation ("how to survive in *extreme* heat", "whats the *fastest* way to purify water"). Survival facts are *lenient*: the question may add one word of context that no phrasing contains. That word is never allowed to be:
- a word some other survival answer is about: "how long is **freezer** food good when the power is out" must not get the fridge answer
- the method, after "with" or "using": "purify water **with a lifestraw**" is not "boil it"
- who it's for, or a number or negation: "my **friend** fell through the ice" needs rescue advice, not self-rescue, and "**baby**", "**dog**" and "**5** gallons" all change the answer

Each of those rules came from a wrong answer found in testing ([below](#survival-questions)).

## How a question is answered

```
"Whats the best way to purify water??"
   │ normalize         lowercase, strip junk            "whats the best way to purify water"
   │ compute           math, money, units, dates,        not computable
   │                   water, bleach, lightning
   │ knowledge gate    indexed match, 97,490 phrasings   fact #3,307  ("what's the best way" = "how")
   │                   no match -> "I don't know."
   │ model (int4 GPT)  fact's shortest phrasing -> answer "how to clean water"
   ▼                   greedy, stops at EOS, 48 chars max
"Boil it for 1 minute (3 above 6,500 ft)."
```

## How it stays direct

Directness is enforced at every layer, not just requested in a prompt:

1. **Data.** Every training answer in `data/facts*.tsv` is the answer only: no preamble, no restated question.
2. **Loss.** The loss covers answer characters only. The model never learns to write anything before the answer.
3. **Stop token.** Each answer ends in EOS, and generation stops there.
4. **Greedy decoding.** It always takes the most likely token: no sampling, no temperature, no rambling.
5. **Hard cap.** An answer is at most 48 characters, enforced in C.
6. **Knowledge gate** (`gate.c`). A tiny model can't tell what it doesn't know. Ask for a capital it never saw and it will invent one. The gate checks the question against every phrasing the model was trained on. It tolerates typos, plurals, split words ("humming bird") and filler, and it requires the question type to agree: a "how many" question never gets a "what" answer. If two different facts match equally well, it refuses rather than guess. With no match, the answer is `I don't know.`
7. **Computed answers** (`calc.c`, `convert.c`, `everyday.c`). Anything that can be calculated is, rather than memorized, and number words work too ("twelve times seven", "half of 30", "a dozen"):
   - math: `12*7`, `15% of 80`, `50 plus 10%` (= 55, like any calculator), `square root of 144`, `average of 3, 5 and 10`
   - money: `15% tip on 42.50` → `Tip 6.38, total 48.88.`, `8% tax on 50`, `20% off 80`, `what percent is 12 of 48`, `percent change from 50 to 75`, `split 90 between 4 people`
   - 63 units: `10 km in miles`, `how many pounds is 70 kg`, `350 f in c`, `32 psi in bar`
   - times: `3pm in 24 hour time`, `how long from 9am to 5:30pm`, `3pm eastern in pacific`, `10:30 utc in jst`
   - dates: `what day was july 20 1969`, `days between march 3 and june 10`, `30 days after march 3 2025`, `is 2100 a leap year`
   - survival: water for N people and days, bleach for N gallons or liters, distance to lightning
   - health: `bmi 70 kg 175 cm` → `BMI 22.9: healthy weight.`
   - numbers: `255 in hex`, `1994 in roman numerals`, `is 91 prime` → `No, 91 = 7 x 13.`, `10 factorial`
   - chance: `flip a coin`, `roll 2 dice`, `random number between 1 and 10` (seeded from the ESP32's hardware RNG)

   When a computed answer would depend on something it can't know, it says `I don't know.` instead of guessing. For example, "3pm eastern in UTC" depends on daylight saving, and it has no clock.

## v2

v1 was the first build: a small int8 model that tried to answer from raw text. v2 is everything below, measured on the C engine:

| | v1 | **v2** |
|---|---|---|
| Facts the model stores | 309 | **24,854** (197 survival) |
| Accepted phrasings | 639 | **97,490** |
| Parameters | 616K | **2.03M** (6 layers, dim 192) |
| Computed questions | arithmetic | **+ money, units, times, time zones, dates, BMI, bases, primes, dice, survival sums** |
| Model file (flash) | 720 KB | **2,974 KB** |
| Weights | int8 | **int4 + fp16 scales + quantization-aware training** |
| Knowledge gate | scans every phrasing | **word index, strict for names, lenient for advice** |
| Matmuls on dual-core chips | 1 core | **2 cores** |
| Trained phrasings correct | 97% | **100% (97,490/97,490)** |
| Held-out questions correct | 85% (58/68) | **99.4% (507/510), 0 wrong** |

### Fitting more knowledge in the same chip
- **Retrieval-canonical prompts.** The gate finds the fact, and the model only ever sees that fact's key, never the raw text. Capacity goes to facts instead of typos, and accuracy on the chip matches training. v1 answered "France capital" with "Beijing."; that class of error is gone.
- **Shortest phrasing as the key.** On the chip every prompt character is a full forward pass, so each fact's key is its shortest accepted phrasing ("gold melting point", not "what is the melting point of gold").
- **int4 weights with quantization-aware training.** 4.5 bits per weight: one fp16 scale per 32 weights. The second half of training uses the rounded weights through a straight-through estimator, and `export.py` asserts its rounding matches training bit for bit.
- **Flash bandwidth is the real bottleneck.** Weights stream from memory-mapped flash through a 32 KB cache. The int4 kernel reads 8 weights per 32-bit load and uses unsigned nibbles with a precomputed-sum correction: Σ(w−8)·x = Σw·x − 8·Σx.
- **Errata for the last 0.02%.** After training, the int4 model answers 24,849 of 24,854 facts exactly on the C engine. `export.py` runs the real engine on every fact and stores the 5 it gets wrong as text: the hummingbird's weight, Carl Sagan, Dr. Seuss, Chandragupta Maurya and one publication year. Continuing training to fix them made things worse (24,850 → 24,811 in Python), because every update flips int4 rounding across the whole network.
- **Grouped-query attention** halves the K/V projections and the KV cache.
- **Both cores.** On the ESP32 and ESP32-S3 each matmul is split across the two cores with FreeRTOS task notifications. Output rows are independent, so answers are bit-identical to single-core.
- **One big app partition.** `firmware/partitions.csv` gives the app 3.9 MB of a 4 MB flash (no OTA), which the 3.0 MB model needs.

### Knowledge from real sources
- **Wikidata, filtered for reliability.** `data/wikidata_facts.py` queries Wikidata (public domain, CC0) through [QLever](https://qlever.dev) and keeps the best-known items by how many Wikipedias cover them: 3,000 people, 1,500 cities, every country, presidents, mountains, rivers, landmarks, books, films, songs, albums, paintings, companies, universities, wars and battles, compounds, species and programming languages. What it throws away matters as much:
  - dates respect Wikidata's precision (a month-precise date never gets a made-up day), and nothing before 1583
  - birthplaces must be towns or cities, not hospitals; quantities must be in the expected unit
  - song and album credits need a single performer; titles must be distinctive ("Crash" could be a dozen films)
  - only organic chemical formulas, where Wikidata's Hill notation is the familiar form (it writes NaCN as "CNNa")
  - hand-written facts always win over Wikidata ones
- **Names must match in full.** Wikidata facts are *strict*: every word of a phrasing must be in the question, so "where was obama born" can't reach Michelle Obama's birthplace. Barack's birthplace didn't pass the filters, and before this rule that question got Michelle's. Short names are added only where one person clearly dominates: "einstein", "lincoln", "trump" and "obama" (Barack, with 2.5x Michelle's Wikipedia coverage) work, but "roosevelt", "bush", "jackson" and "curie" don't. A name the gate can only partly see is dropped entirely: "Will Smith" is just "smith" to it, since "will" is a filler word.
- **Datasets, not memory.** Every chemical element comes from [mendeleev](https://github.com/lmmentel/mendeleev) (MIT). Currencies come from Unicode CLDR via Babel (BSD), calling codes from phonenumbers (Apache 2.0), US and Canadian abbreviations from pycountry (LGPL), and 20 CODATA physical constants from SciPy (BSD).
- **Everyday knowledge from standard references:** safe cooking temperatures and food storage (USDA / FoodSafety.gov), emergency numbers for 18 countries and the EU, first aid (Red Cross / NHS), health reference numbers (CDC / AHA / FDA / ADA), and home and car settings (US DOE).

## Specs

| | |
|---|---|
| Architecture | GPT: 6 layers, dim 192, 4 query heads / 2 K/V heads, MLP 576, context 96, tied embeddings |
| Parameters | 2,030,208 |
| Knowledge | 24,854 facts, 97,490 accepted phrasings, plus computed answers |
| Model in flash | 2,974 KB: int4 weights 1,094 KB, int8 embeddings 37 KB, fact keys 621 KB, gate index 1,213 KB, errata 188 bytes |
| RAM at runtime | about 125 KB: int8 KV cache 113 KB, activations and buffers about 12 KB (199 KB of heap still free on an ESP32) |
| Engine code | pure C99, no dependencies |
| Works on | ESP32, ESP32-S3, ESP32-C3 (4 MB flash) |

## Results

Scored by `train/evaluate.py`, which runs the real C engine (the same code the ESP32 runs), not the Python model:

| Test | Score |
|---|---|
| Every trained phrasing (97,490 questions the gate was built from) | 97,490 / 97,490 |
| Every fact's key, answered by the model alone (no errata) | 24,849 / 24,854 |
| test questions it never trained on | 507 / 510, and the 3 misses are `I don't know.` |
| Gate index parses like the C gate | 97,485 / 97,485 |
| Answers over 48 chars or starting with filler | 0 |

Speed on one laptop CPU core: about 48 ms per model answer, under 1 ms to refuse, and effectively instant for computed answers.

### Test questions it never trained on

`data/eval.tsv` holds 510 questions that are not in the training data. Each is typed as a person would (missing apostrophes, no capitals, extra words, typos) and has one exact expected answer. The C engine must produce that answer character for character: a close answer counts as wrong.

| Group | What it checks | Score |
|---|---|---|
| Rephrased general facts | the same fact asked differently: "France capital", "hummingbird weight", "who painted The Scream?" | 83 / 83 |
| ...look-alike traps inside that group | picking the right one of two similar facts: Iceland vs Ireland, Guinea vs Guinea-Bissau, Niger vs Nigeria, Fahrenheit→Celsius vs Celsius→Fahrenheit | all correct |
| Calculator and conversions | "(3+4)*5", "1/0", "sqrt 2", "10 km in miles", "1994 in roman numerals", "is 91 prime" | 22 / 22 |
| Must refuse | no fact behind them, so `I don't know.` rather than a guess: "how many moons does jupiter have", "how far is pluto" | 19 / 19 |
| Everyday, batch 1 | "whats the safe temp for chicken", "15% tip on 42.50", "my wifi isn't working", plus 10 that must be refused | 90 / 90 |
| Everyday, batch 2 | 85 more, written after batch 1 was tuned | 85 / 85 |
| Wikidata knowledge | "einstein birthday", "who directed the godfather", "how high is kilimanjaro", plus "who was roosevelt" and "who was bush", which must be refused as ambiguous | 40 / 40 |
| Survival, batch 1 | "can i eat snow if im thirsty", "a snake bit me", "how do i get out of a rip current", plus 9 that must be refused | 69 / 69 |
| Survival, batch 2 | "whats the fastest way to purify water", "a moose is charging at me", plus 8 that must be refused | 55 / 55 |
| Survival, batch 3 | "how do i scare off a mountain lion", "lightning struck 15 seconds before the thunder", plus 8 that must be refused ("what do i do if my friend falls through the ice", "how do i purify water with a lifestraw") | 44 / 47 |
| **Total** | | **507 / 510** |

By kind of answer, across all groups: 348 answered from stored knowledge, 91 computed, 68 correctly refused. The 3 misses are all `I don't know.`: "is cotton clothing bad for camping", "how many drops of bleach to clean a gallon of water", "which knot joins two ropes of different sizes".

#### Survival questions

The survival tests were written in three batches, each *before* looking at how the gate handled it, and each first score is recorded here as it was:

| Batch | First run, before any change | What was fixed afterwards |
|---|---|---|
| 1: 69 questions | 34 right, **0 wrong**, 33 refused | synonyms (frostbitten, forest, elevation, "no matches"), more phrasings |
| 2: 55 questions | 22 right, **0 wrong**, 33 refused | lenient advice facts; "what's the best way to" = "how" |
| 3: 47 questions | 23 right, **5 wrong**, 19 refused | the leniency rules above: freezer vs fridge, "with a flint", "my friend", "how can I **tell** if" |

The honest takeaway: **on survival questions worded in a way it hasn't seen, expect it to answer about half and decline the rest.** Batch 3 is the warning: the first version of the leniency rule gave 5 wrong answers in 47 questions. After the fixes, all 510 test questions give either the right answer or `I don't know.`, but a new batch could find a new way to be wrong. Keep questions short and plain ("signs of hypothermia", "how to purify water") for the best results.

**What "never trained on" means, precisely.** The model never saw any of these questions. The knowledge gate was adjusted while looking at the results of each batch, so after its fixes each batch is a regression test, not an unbiased measure. The first-run scores above are the unbiased ones.

**How it can still be wrong:**
- **Unusual phrasing** can match a similar but different fact. The gate checks words, question type and coverage, but it matches words, not meaning.
- **Facts can be outdated or simplified.** Populations, leaders and guidelines change, and a 48-character answer leaves out detail. Some answers are US-specific (gallons, 911) and say so where it matters.
- **Survival advice is general.** Local conditions, species and injuries vary. A short answer is a reminder, not training.
- **Speed on a physical board hasn't been measured,** only correctness in the emulator.

### Verified on ESP32 firmware

- **Builds** with PlatformIO for all three targets with no warnings:

  | Board | Flash used (of the 3.9 MB app partition) | Static RAM |
  |---|---|---|
  | ESP32 | 3,396 KB (82.3%) | 33.7 KB |
  | ESP32-S3 | 3,393 KB (82.2%) | 30.7 KB |
  | ESP32-C3 | 3,378 KB (81.8%) | 26.0 KB |

- **Runs** in Espressif's ESP32 emulator (QEMU) as the real flash image, on both cores: it boots with 199 KB of heap free and answers all 510 test questions plus a random 300 of the trained phrasings exactly like the PC build (807 / 810 as expected; the 3 are the refusals above).
- **Not yet timed on physical hardware.** Emulator timings aren't real. Each answer streams the 1.1 MB of int4 weights from flash once per character, so a rough, unmeasured estimate is a few seconds per model answer; computed answers and refusals take milliseconds. To measure on a board, set `SHOW_TIMING 1` in `firmware/src/main.cpp`.

Re-run the emulator check yourself (needs [Espressif's QEMU](https://github.com/espressif/qemu/releases)):

```sh
cd firmware && pio run -e esp32dev
echo "how do i purify water" | python qemu_test.py --qemu path/to/qemu-system-xtensa
```

## Layout

```
data/facts.tsv              hand-written everyday facts
data/facts_survival.tsv     hand-written survival facts, with sources (edit these)
data/more_facts.py          dataset facts -> data/facts_generated.tsv
data/wikidata_facts.py      Wikidata facts -> data/facts_wikidata.tsv
data/eval.tsv               held-out questions: paraphrases, everyday, survival, must-refuse
train/common.py             tokenizer, normalization, fact loading (mirrored in C)
train/model.py              the transformer, int4 fake-quantization for QAT
train/train.py              training (resumable: --resume)
train/export.py             int4 quantization + export
train/gate_index.py         builds the gate's word index (mirrors gate.c)
train/check_routing.py      fast check: every phrasing reaches its fact
train/evaluate.py           scores the real C engine
firmware/lib/tinyai/        inference engine, gate, calculator, units, everyday and survival math (C99)
firmware/src/main.cpp       serial chat for the ESP32
firmware/partitions.csv     one 3.9 MB app partition for the model
firmware/qemu_test.py       runs the built firmware in the ESP32 emulator
host/                       PC build of the same engine
```

## Limits

- It knows what's in `data/facts*.tsv` and nothing else. Anything else gets `I don't know.`, which is the point.
- No clock, GPS or internet: "what time is it", "where am I", "weather" and "news" can't be answered.
- First-aid, health and survival answers are short reference facts, not medical advice or training.
- The gate matches words, not meaning. Heavily reworded questions can be refused even when the fact is known. Add those phrasings to the fact's line.
- 2M parameters can store facts. They can't reason. Multi-step questions are out of scope.
- Numbers from datasets are as good as the datasets: Wikidata as queried in September 2026, element data from mendeleev, currencies from CLDR as of Babel 2.18.
