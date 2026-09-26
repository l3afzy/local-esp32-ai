# local-esp32-ai

An offline survival and everyday assistant that runs entirely on an ESP32.
No Wi-Fi, no cloud, no signal needed. It answers directly: the answer, nothing else.

> **It might make a mistake.** Double-check anything that matters. It is not a substitute for first-aid or survival training: **in an emergency, call for help first** if you can.

```
you: how do i purify water
esp: Boil it for 1 minute (3 above 6,500 ft).
you: what do i do if i see a bear
esp: Stay calm, don't run; talk and back away slowly.
you: signs of hypothermia
esp: Shivering, confusion, slurred speech, drowsy.
you: thunder 10 seconds after lightning
esp: About 2.1 miles (3.4 km). Go indoors.
you: how much does a hummingbird weigh
esp: Between 2 and 20 grams.
you: how do i build a raft
esp: I don't know.
```

## What it knows

- **Survival:** 197 facts on water, fire, shelter, signaling, navigation, food, cold, heat, wildlife, weather, first aid and knots, from CDC, NWS, National Park Service, Ready.gov, Red Cross, NHS, WHO and the US Army survival manual.
- **General knowledge:** 21,369 facts from Wikidata (people, places, books, films, science) and about 3,300 more from datasets and hand-written lists.
- **Calculated, not memorized:** math, money, units, times, dates, water for N people, bleach doses, lightning distance.

Anything it doesn't know gets `I don't know.` instead of a guess.

## Flash it

You need [PlatformIO](https://platformio.org/install/cli) and an ESP32 board with 4 MB of flash (ESP32, ESP32-S3 or ESP32-C3).

```sh
cd firmware
pio run -t upload            # or: pio run -e esp32-s3 -t upload
pio device monitor           # type questions, press Enter
```

Or try it on a PC first, with the same C engine:

```sh
make -C host && ./host/tinyai firmware/data/model.bin
```

## Results

| | |
|---|---|
| Trained questions answered exactly | 97,490 / 97,490 |
| Test questions it never trained on | 507 / 510 (the 3 misses say `I don't know.`) |
| Real firmware in the ESP32 emulator | 807 / 810 |

On survival questions worded in ways it hasn't seen, expect it to answer about half and refuse the rest. Short, plain questions work best.

## Train your own

```sh
pip install -r requirements.txt
python train/train.py          # about 3 hours on a 4-core CPU, no GPU
make -C host && python train/export.py
python train/evaluate.py
```

To add facts, add a line to `data/facts.tsv` or `data/facts_survival.tsv` and retrain:

```
question|another phrasing<TAB>Direct answer.
```

## Specs

2M-parameter GPT (6 layers, int4 weights), a 3.0 MB model file, about 125 KB of RAM. Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) curriculum.

How it works, how it was tested, and its limits: [docs/details.md](docs/details.md).
