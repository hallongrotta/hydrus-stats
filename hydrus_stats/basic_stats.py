import json
from typing import Any, Dict, List, Tuple

from hydrus_stats.utils import get_tags


def get_counts(metadata: Dict[str, Dict[str, Any]], min_count: int, exclude_namespaced: bool =False, tags_to_exclude: List[str]=None):
    """
    Create tag counts.
    :param metadata: List of metadata dictionaries.
    :param min_count: Minimum count for a tag to be included in the vocabulary.
    :param exclude_namespaced: Whether to exclude namespaced tags.
    :param tags_to_exclude: List of tags to exclude.
    :return: Dictionary of tag counts.
    """
    tag_counts: Dict[str, int] = {}

    for entry in metadata:
        for tag in get_tags(entry, exclude_namespaced):
            try:
                tag_counts[tag] += 1
            except KeyError:
                tag_counts[tag] = 1

    # Filter out tags that occur less than min_count times
    keys = list(tag_counts.keys())
    for t in keys:
        if tag_counts[t] < min_count:
            del tag_counts[t]

        if tags_to_exclude is not None and t in tags_to_exclude:
            del tag_counts[t]

    return tag_counts


def generate_counts(metadata, min_count=2, exclude_namespaced=True, tags_to_exclude=None, outfile=None):
    """Create tag counts."""
    tag_counts = get_counts(
        metadata, min_count, exclude_namespaced, tags_to_exclude
    )

    data_dict = {}

    num_documents = len(metadata)
    num_tags = len(tag_counts)

    data_dict["num_documents"] = num_documents
    data_dict["num_tags"] = num_tags
    data_dict["counts"] = tag_counts

    if outfile is not None:
        with open(outfile, "w", encoding="utf8") as f:
            json.dump(data_dict, f)
        print(f"Saved counts file to {outfile}.")

    return tag_counts, num_documents, num_tags


def generate_vocab(tag_counts: Dict[str, int], min_count: int, outfile: str =None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary.
    :param tag_counts: Dictionary of tag counts.
    :param min_count: Minimum count for a tag to be included in the vocabulary.
    :param outfile: Path to output file.
    :return: Dictionary of tag indices and tag strings.
    """
    tag2idx: Dict[str, int] = {}
    idx2tag: Dict[int, str] = {}
    i = 0

    for key, count in tag_counts.items():
        if count < min_count:
            continue
        tag2idx[key] = i
        idx2tag[i] = key
        i += 1
    
    if outfile is not None:
        with open(outfile, "w", encoding="utf8") as f:
            for key in tag2idx:
                f.write(key + "\n")
        print(f"Saved vocab file to {outfile}.")

    return tag2idx, idx2tag
