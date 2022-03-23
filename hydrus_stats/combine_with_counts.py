from audioop import add
from tqdm import tqdm
import pandas as pd


def add_counts():
    """Add tag counts to PMI pairs"""
    counts = pd.read_json("stats_full//counts.json")

    tag_pairs = pd.read_csv("stats_full//tag_mi_filtered.csv", names=["tag1", "tag2", "mi"], keep_default_na=False)

    sum_counts = []
    pbar = tqdm(total=len(tag_pairs))
    for i, row in tag_pairs.iterrows():

        sum_counts.append(counts.loc[row["tag1"]]["counts"] + counts.loc[row["tag2"]]["counts"])
        pbar.update()

    tag_pairs = tag_pairs.assign(sum_counts=sum_counts)

    tag_pairs = tag_pairs.sort_values(by=["mi", "sum_counts"], ascending=False)

    tag_pairs.to_csv("stats_full//tag_mi_filtered_counts.csv", index=False)


if __name__ == "__main__":
    add_counts()