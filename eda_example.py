#!/usr/bin/env python3
"""
Quick exploratory data analysis (EDA) script.

Usage examples:
  python eda_example.py --data tips
  python eda_example.py --data path/to/file.csv

Outputs textual summaries to stdout and writes plots into the `plots/` folder.
"""
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def summarize(df):
    print("Shape:", df.shape)
    print("\nHead:\n", df.head().to_string(index=False))
    print("\nDescribe:\n", df.describe(include='all').T)
    print("\nMissing values:\n", df.isnull().sum())


def feature_engineer(df: pd.DataFrame, out_dir: str = "plots", save_path: Optional[str] = None) -> pd.DataFrame:
    """Perform light, general-purpose feature engineering and return a new dataframe.

    Steps performed:
    - simple imputation (median for numeric, mode for categorical)
    - log1p transform for highly skewed positive numeric features
    - interaction feature (product of two highest-variance numeric cols)
    - quantile binning for numeric features
    - one-hot encoding for categorical features
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Impute numeric with median
    for c in num_cols:
        if df[c].isnull().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # Impute categorical with mode
    for c in cat_cols:
        if df[c].isnull().any():
            try:
                mode = df[c].mode(dropna=True)[0]
            except Exception:
                mode = ""
            df[c] = df[c].fillna(mode)

    created = []

    # Log transform for skewed numeric features (positive values only)
    if num_cols:
        skews = df[num_cols].skew().abs()
        for col, sk in skews.items():
            if sk > 1.0 and (df[col] > 0).all():
                new_name = f"{col}_log1p"
                df[new_name] = np.log1p(df[col])
                created.append(new_name)

    # Interaction: product of two numeric cols with largest variance
    if len(num_cols) >= 2:
        variances = df[num_cols].var().sort_values(ascending=False)
        top = variances.index[:2].tolist()
        if len(top) == 2:
            inter_name = f"{top[0]}_x_{top[1]}"
            df[inter_name] = df[top[0]] * df[top[1]]
            created.append(inter_name)

    # Quantile binning
    for c in num_cols:
        try:
            bname = f"{c}_qbin"
            df[bname] = pd.qcut(df[c], q=4, duplicates='drop')
            created.append(bname)
        except Exception:
            continue

    # One-hot encode categorical columns (drop first to avoid collinearity)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, drop_first=True)

    # Save engineered dataset if requested
    ensure_dir(out_dir)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Engineered dataset saved to {save_path}")
    else:
        eng_path = os.path.join(out_dir, "engineered.csv")
        df.to_csv(eng_path, index=False)
        print(f"Engineered dataset saved to {eng_path}")

    if created:
        print("Created features:", created)
    else:
        print("No additional features created.")

    return df


def main():
    p = argparse.ArgumentParser(description="Quick EDA")
    p.add_argument("--data", "-d", help="CSV path or 'tips' for example dataset", default="tips")
    p.add_argument("--out", "-o", help="Output folder for plots", default="plots")
    p.add_argument("--feature-engineer", "-f", help="Run feature engineering and save engineered dataset", action='store_true')
    p.add_argument("--save-engineered", help="Path to save engineered CSV (optional)", default="")
    args = p.parse_args()

    if args.data == "tips":
        df = sns.load_dataset("tips")
    else:
        df = pd.read_csv(args.data)

    ensure_dir(args.out)
    summarize(df)

    if args.feature_engineer:
        save_path = args.save_engineered if args.save_engineered else None
        df = feature_engineer(df, out_dir=args.out, save_path=save_path)

    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Correlation heatmap for numeric columns
    if not num.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="vlag")
        plt.title("Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "correlation_heatmap.png"))
        plt.close()

        # Pairplot (saved via the returned PairGrid)
        try:
            g = sns.pairplot(num)
            g.savefig(os.path.join(args.out, "pairplot.png"))
            plt.close()
        except Exception:
            pass

    # Distributions for numeric columns
    for col in num.columns:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"dist_{col}.png"))
        plt.close()

    # Countplots for categorical columns
    for c in cat:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=c)
        plt.title(f"Counts: {c}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"count_{c}.png"))
        plt.close()

    # Boxplots of numeric by categorical
    if len(cat) > 0 and not num.empty:
        for n in num.columns:
            for c in cat:
                try:
                    plt.figure(figsize=(8, 4))
                    sns.boxplot(data=df, x=c, y=n)
                    plt.title(f"{n} by {c}")
                    plt.tight_layout()
                    safe_name = f"box_{n}_by_{c}.png".replace(" ", "_")
                    plt.savefig(os.path.join(args.out, safe_name))
                    plt.close()
                except Exception:
                    continue

    print("Plots saved to", args.out)


if __name__ == '__main__':
    main()
