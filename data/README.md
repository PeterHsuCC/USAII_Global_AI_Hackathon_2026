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
