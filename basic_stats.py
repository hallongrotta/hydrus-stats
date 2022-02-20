import json

from hydrus_stats.utils import get_tags


def get_counts(metadata, min_count, exclude_namespaced=False, tags_to_exclude=None):
    tag_counts = dict()

    for entry in metadata:
        for t in get_tags(entry, exclude_namespaced):
            try:
                tag_counts[t] += 1
            except KeyError:
                tag_counts[t] = 1

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


def generate_vocab(tag_counts: dict, min_count, outfile=None):
    """Create an ordered vocab."""
    tag2idx = {}
    idx2tag = {}
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
