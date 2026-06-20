# data/

Local datasets for the training scripts in `scripts/`. Everything in this
directory except this file is gitignored -- download datasets yourself
and place them here.

## Cyberbullying Classification (Kaggle, andrewmvd)

https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

Download `cyberbullying_tweets.csv` and place it at:

```
data/cyberbullying_tweets.csv
```

Then run:

```bash
python scripts/train_cyberbullying.py --cyberbullying-csv data/cyberbullying_tweets.csv --max-examples 500
```

## PAN12 / VTPAN (Grooming Head)

PAN12 is the ground truth corpus for the "Sexual Predator Identification" task at
PAN 2012 / CLEF 2012. Unlike the Kaggle dataset above, no application or account is
required:

https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html
(mirrored on Zenodo: https://zenodo.org/record/3713280)

ChatCoder2 and the PANC dataset derived from it are **not used by this project**.
ChatCoder2 requires a manual access application to the dataset maintainer, which
this project did not pursue; PANC is built from ChatCoder2 positive chats plus
PAN12 negative chats, so it cannot be generated without ChatCoder2 either (see
`Hybrid_Conversation_Risk_Detection_Report_v15.docx`, Section 19.1). Grooming Head
training (`scripts/train_grooming.py`) uses PAN12 and its VTPAN subset only.

### 1. Download and place the raw PAN12 corpus

From the PAN12 archive(s), place these files at `data/PAN12/raw_dataset/`:

```
data/PAN12/raw_dataset/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
data/PAN12/raw_dataset/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
data/PAN12/raw_dataset/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml
data/PAN12/raw_dataset/pan12-sexual-predator-identification-groundtruth-problem1.txt
data/PAN12/raw_dataset/pan12-sexual-predator-identification-groundtruth-problem2.txt
```

`data/PAN12/create_datapack.py`, `data/util.py`, and `data/DynamicArray.py` (ported
from the eSPD-datasets pipeline, MIT licensed) convert these into the JSON
"datapack" format the loader consumes. They must be run with `data/` as the
working directory, since the script's paths are relative to it:

```bash
cd data && python PAN12/create_datapack.py
```

This writes `data/PAN12/datapacks/datapack-PAN12-train.json` and `-test.json`.

### 2. VTPAN (curated PAN12 subset)

`data/VTPAN/VTPAN_train_IDs.txt` and `VTPAN_test_IDs.txt` (already in this repo)
list the conversation ids VTPAN keeps from PAN12. From the repo root:

```bash
python data/VTPAN/create_datapack.py
```

This writes `data/VTPAN/datapacks/datapack-VTPAN-train.json` and `-test.json`.
VTPAN is a curated *subset* of PAN12, not an independent dataset -- it's used as a
supplementary, cleaned evaluation split (Section 19.1), not pooled into training.

### 3. Train

```bash
python scripts/train_grooming.py --variant all --max-conversations 50   # smoke test first
python scripts/train_grooming.py --variant all                          # full run
```

See `python scripts/train_grooming.py --help` for the A/B/C ablation variants,
LLM-signal caching (`--llm-cache`, variant C only, needs `ANTHROPIC_API_KEY`), and
the optional `--vtpan-dir` supplementary evaluation.
