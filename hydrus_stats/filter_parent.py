import csv
import pandas as pd
from tqdm import tqdm

# Read parents from .txt files exported from hydrus and filter out existing 
# parents from MI pairs.

parents = pd.read_csv("tag_parents.csv", header=0)
parents = parents.set_index(keys=["parent_tag", "child_tag"])


def is_parent(parent: str, child: str) -> bool:
    """
    Check if parent is parent of child.
    :param parent: parent tag
    :param child: child tag
    :return: True if parent is parent of child, False otherwise.
    """
    try:
        _ = parents.loc[parent, child]
        return True
    except KeyError:
        return False


def main() -> None:
    """Main function. """
    with open("stats_full/tag_mi.csv", "r", encoding="utf8") as infile, open(
        "stats_full/tag_mi_filtered.csv", "w", encoding="utf8", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for tag1, tag2, mi in tqdm(reader):
            if not (is_parent(tag1, tag2) or is_parent(tag2, tag1)):
                writer.writerow((tag1, tag2, mi))
                continue

            #tqdm.write(f"Removed {tag1}, {tag2}.")

if __name__ == "__main__":
    main()
