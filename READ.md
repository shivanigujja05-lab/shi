# Quick EDA example

This workspace adds a small example script to run a quick exploratory data analysis using `pandas` and `seaborn`.

Files added:
- eda_example.py - runnable script that loads `tips` dataset or a CSV and saves plots to `plots/`.
- requirements.txt - Python dependencies.

Run:

```bash
python -m pip install -r requirements.txt
python eda_example.py --data tips
```

To use your own CSV:

```bash
python eda_example.py --data path/to/your.csv
```
