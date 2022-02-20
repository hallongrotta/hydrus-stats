import threading
from queue import Queue

import numpy as np
from tqdm import tqdm

from .utils import get_tags


def sum_tf_idfs(list_of_tags):
    sum = 0
    for _, v in list_of_tags:
        sum += v
    return sum


def calc_idf(num_documents, tag_counts):
    return {k: np.log(num_documents / v) for k, v in tag_counts.items()}


class TFIDF:
    def __init__(self, tag_vocab, tag_counts, num_documents) -> None:
        self.tag_vocab = tag_vocab
        self.tag_counts = tag_counts
        self.num_documents = num_documents
        self._idf = None

        self.metadata_queue: Queue = Queue()
        self.tf_idf_queue: Queue = Queue()

    @property
    def idf(self):
        if self._idf is None:
            idf = calc_idf(self.num_documents, self.tag_counts)
            self._idf = idf

        return self._idf

    def tf_idf_multithread(self, metadata, filter_namespaced):
        num_threads = 6
        threads = []
        vectors = {}

        for entry in metadata:
            self.metadata_queue.put(entry)
        
        thread: threading.Thread
        for _ in range(num_threads):
            thread = threading.Thread(target=self.tf_idf_worker, daemon=True, kwargs={'filter_namespaced': filter_namespaced})
            thread.start()
            threads.append(thread)

        self.metadata_queue.join()

        for thread in threads:
            thread.join()

        while not self.tf_idf_queue.empty():
            img_hash, data = self.tf_idf_queue.get()
            vectors[img_hash] = data


        return vectors

    def tf_idf_worker(self, filter_namespaced=False):
        while True:
            entry = self.metadata_queue.get()
            img_hash = entry["hash"]
            tags = get_tags(entry, filter_namespaced)
            if len(tags) != 0:
                self.tf_idf_queue.put((img_hash, self.tf_idf(tags)))

    def tf_idf(self, tags):
        tf_idfs = []
        tf = 1 / len(tags)
        for i, tag in enumerate(tags):
            try:
                tf_idfs.append((self.tag_vocab[tag], self.idf[tag] * tf))
            except KeyError:
                continue
        return tf_idfs

    def get_tf_idf_vectors(self, metadata, filter_namespaced=False):
        vectors = {}
        pbar = tqdm(total=len(metadata), desc="Calculating TF-IDF")
        for entry in metadata:
            img_hash = entry["hash"]
            tags = get_tags(entry, filter_namespaced)
            if len(tags) == 0:
                continue

            vectors[img_hash] = self.tf_idf(tags)
            pbar.update()

        return vectors


def sort_images_by_tf_idf(metadata, tag_vocab, tag_counts, num_documents, outfile):
    tf_idf: TFIDF = TFIDF(tag_vocab, tag_counts, num_documents)

    vectors = tf_idf.get_tf_idf_vectors(metadata, filter_namespaced=True)

    sorted_vects = sorted(vectors.items(), key=lambda x: sum_tf_idfs(x[1]))

    with open(outfile, "w", encoding="utf8") as f:
        for h, _ in sorted_vects:
            f.write(h + "\n")
    print(f"Saved sorted hashes to {outfile}")