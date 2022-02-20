
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import hydrus_api.utils

from hydrus_stats.basic_stats import generate_counts, generate_vocab
from hydrus_stats.mutual_information import (calculate_cooccurrences,
                                             calculate_mututal_information,
                                             sort_tag_pairs_by_pmi)
from hydrus_stats.tf_idf import sort_images_by_tf_idf
from hydrus_stats.utils import load_counts_from_file


def get_data_from_hydrus(url, api_key, tags_to_search):
    """Query hydrus for tags."""
    client: hydrus_api.utils.Client = hydrus_api.Client(api_key, api_url=url)
    try:
        all_file_ids = client.search_files(tags_to_search)
    except hydrus_api.ConnectionError:
        print("Could not connect to hydrus.")
        exit(1)

    metadata = []
    for ids in hydrus_api.utils.yield_chunks(all_file_ids, 10):
        metadata.extend(client.get_file_metadata(file_ids=ids))

    return metadata


def parse_args():
    """Parse CLI arguments."""
    parser = ArgumentParser()
    parser.add_argument("--url", type=str, help="URL to access the client API.")
    parser.add_argument("--api-key", type=str, help="API key for client API.")
    parser.add_argument("--output", type=str, default="stats", help="Path to folder where output files will be saved.")
    subparsers = parser.add_subparsers(help='sub-command help', dest="task")
    mi_parser = subparsers.add_parser("mi")
    mi_parser.add_argument("--counts", type=str, help="Use the given counts file instead of calculating one.")
    mi_parser.add_argument("--min-count", type=int, default=3, help="Minumum number of times a tag has to occur to be included in calculations.")
    mi_parser.add_argument("--cooccurrences", type=str, help="Use the given cooccurrence file instead of calculating one.")
    mi_parser.add_argument("--mi-file", type=str, help="Use the given PMI file instead of calculating one.")
    mi_parser.add_argument("query", type=str, nargs='+')

    tf_idf_parser = subparsers.add_parser("tfidf")
    tf_idf_parser.add_argument("--counts", type=str, help="Use the given counts file instead of calculating one.")
    tf_idf_parser.add_argument("--min-count", type=int, default=3, help="Minumum number of times a tag has to occur to be included in calculations.")
    tf_idf_parser.add_argument("query", type=str, nargs='+')

    args = parser.parse_args()

    return args

def main():
    """Main function."""
    args = parse_args()

    if args.query is not None:
        metadata = get_data_from_hydrus(args.url, args.api_key, tags_to_search=args.query)
    else:
        metadata = None

    stats_dir = Path(args.output)

    if not stats_dir.is_dir():
        stats_dir.mkdir()

    counts_path = stats_dir / "counts.json"
    if args.counts is not None:
        tag_counts, num_documents, _ = load_counts_from_file(args.counts)
    else:
        if metadata is None:
            raise Exception("Can not count tags from an empty query!")
        tag_counts, num_documents, _ = generate_counts(metadata, min_count=args.min_count, outfile=counts_path)


    vocab_path = stats_dir / "vocab.txt"
    tag2idx, idx2tag = generate_vocab(tag_counts, 0, outfile=vocab_path)


    if args.task == "tfidf":
        tf_idf_hashes = stats_dir / "hashes.txt"
        sort_images_by_tf_idf(metadata, tag2idx, tag_counts, num_documents, tf_idf_hashes)
    elif args.task == "mi":

        if args.cooccurrences is not None:
            with open(args.cooccurrences, "rb") as f:
                cooccurrences = pickle.load(f)
        else:
            cooccurrence_file = stats_dir / "cooccurrences.pickle"
            if metadata is None:
                raise Exception("Can not count cooccurrences from an empty query!")
            cooccurrences = calculate_cooccurrences(metadata, tag2idx, outfile=cooccurrence_file)


        mi_file = stats_dir / "mi.pickle"
        calculate_mututal_information(cooccurrences, tag_counts, idx2tag, num_documents, outfile=mi_file)

        mi_tag_pair_file = stats_dir / "tag_mi.csv"

        with open(mi_file, "rb") as f:
            m_i = pickle.load(f)

        sort_tag_pairs_by_pmi(m_i, idx2tag, mi_tag_pair_file)


    else:
        sys.exit()

if __name__ == "__main__":
    main()