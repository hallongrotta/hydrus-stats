import csv
import pickle

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, tril
from tqdm import tqdm

from hydrus_stats.utils import get_tags


def sort_tag_pairs(m_i: csr_matrix, idx2tag: dict, tag_counts: dict, outfile=None):
    """Sort tag pairs by their PMI."""

    m_i = tril(m_i, format="csr")
    m_i.eliminate_zeros()

    tag_pairs = []

    for i, row in enumerate(m_i):
        tag_i = idx2tag[i]
        i_count = tag_counts[tag_i]
        for j, pmi in zip(row.indices, row.data):
            tag_j = idx2tag[j]
            j_count = tag_counts[tag_j]
            tag_pairs.append((tag_i, idx2tag[j], pmi, i_count + j_count))

    tag_pairs = sorted(tag_pairs, key=lambda x: (-x[2], -x[3]))
    
    if outfile is not None:
        with open(outfile, "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            for entry in tag_pairs:
                writer.writerow(entry)
        print(f"Saved tag pairs to {outfile}")

    return tag_pairs

def calculate_cooccurrences(metadata, vocab, outfile=None):
    """Calculate cooccurrence matrix."""
    vocab_size = len(vocab)
    cooccurrences = lil_matrix((vocab_size, vocab_size), dtype=np.uint32)
    pbar = tqdm(total=len(metadata), desc="Calculating cooccurrences")
    for entry in metadata:
        tag_set = get_tags(entry, exclude_namespaced=True)
        length = len(tag_set)
        for i in range(length-1):
            w_i = tag_set[i]
            w_i_idx = vocab.get(w_i)
            if w_i_idx is None:
                continue
            for j in range(i+1, length):
                w_j = tag_set[j]
                w_j_idx = vocab.get(w_j)
                if w_j_idx is None:
                    continue

                cooccurrences[w_i_idx, w_j_idx] += 1
                cooccurrences[w_j_idx, w_i_idx] += 1
        pbar.update()

    cooccurrences = cooccurrences.tocsr()
    cooccurrences.eliminate_zeros()

    if outfile is not None:
        with open(outfile, "wb") as f:
            pickle.dump(cooccurrences, f)

    return cooccurrences

def calculate_mi(cooccurrences: csr_matrix, counts: np.array, num_documents, normalize=True):
    """Calculate PMI for tag pairs."""
    MI = lil_matrix(cooccurrences.shape, dtype=np.float64)
    pbar = tqdm(total=cooccurrences.shape[0], desc="Calculating mutual information") # Initialise
    for x, row in enumerate(cooccurrences):
        p_x = counts[x]/num_documents
        for y, cooccurrence in zip(row.indices, row.data):
            p_y = counts[y]/num_documents
            p_xy = cooccurrence/num_documents
            m_i = np.log(p_xy/(p_x*p_y))

            if normalize:
                if p_xy != 1.0:
                    m_i = m_i / -np.log(p_xy)



            MI[x, y] = m_i
        pbar.update()
    
    MI = MI.tocsr()
    MI.eliminate_zeros()
    return MI


def calculate_mututal_information(cooccurrences: csr_matrix, tag_counts, idx2tag, num_documents, outfile=None):
    """Calculate PMI for tag pairs, and save to file."""
    vocab_size = len(tag_counts)
    counts_vec = np.zeros((vocab_size), dtype=np.uint32)
    for i in range(vocab_size):
        tag = idx2tag[i]
        counts_vec[i] = tag_counts[tag]

    MI = calculate_mi(cooccurrences, counts_vec, num_documents)

    if outfile is not None:
        with open(outfile, "wb") as f:
            pickle.dump(MI, f)
        print(f"Saved MI data to {outfile}")

    return MI


# graph = nx.Graph()
# for row in df.itertuples():
#    graph.add_edge(row.tag1, row.tag2, weight=row.MI)

# plt.figure(figsize=(40, 40))
# pos = nx.layout.spring_layout(graph, center=(0, 0), scale=100, seed=5, iterations=100)
# nx.draw_networkx_nodes(graph, pos=pos)
# nx.draw_networkx_labels(graph, pos=pos)
# plt.savefig("graph.png")
