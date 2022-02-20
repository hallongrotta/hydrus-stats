import os
import pickle
import webbrowser
from io import BytesIO

import hydrus.utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from hydrus_stats.utils import get_tags, idToURL


def load_glove(embedding_file, vocab_file):

    if os.path.exists("embedding_dict.pickle"):
        with open("embedding_dict.pickle", "rb") as f:
            vocab_embedding = pickle.load(f)
    else:
        vocab = []
        embeddings = torch.load(embedding_file)
        vocab_embedding = dict()
        with open(vocab_file) as f:
            for line in f:
                vocab.append(line.strip())
        for i in range(len(vocab)):
            vocab_embedding[vocab[i]] = (
                embeddings(torch.LongTensor([i + 4])).detach().numpy()
            )

        with open("embedding_dict.pickle", "wb") as f:
            pickle.dump(vocab_embedding, f)

    return vocab_embedding


def glove_images(client: hydrus.Client, glove_embeddings, metadata):
    file_ids = []
    tag_embeddings = []
    image_tags = []

    for entry in metadata:
        embeddings = []
        tags = get_tags(entry)
        filtered_tags = []
        for t in tags:
            try:
                tag_embedding = glove_embeddings[t]
                embeddings.append(tag_embedding)
                filtered_tags.append(t)
            except KeyError:
                continue
        embeddings = np.array(embeddings).reshape(-1, 300)
        aggregate_embedding = embeddings.max(axis=0)
        file_ids.append(entry["file_id"])
        tag_embeddings.append(aggregate_embedding)
        image_tags.append(filtered_tags)

    matrix = np.array(tag_embeddings)

    return matrix, file_ids, image_tags


def dbscan(data, eps):
    dbscan = DBSCAN(eps=eps, metric="euclidean")
    clusters = dbscan.fit_predict(data)
    return clusters


def plot_pca(matrix, file_ids, image_tags, num_to_plot=100):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix)

    num_to_plot = 1000
    X = reduced[:num_to_plot, 0]
    Y = reduced[:num_to_plot, 1]

    num_clusters = np.zeros(10)
    x = np.linspace(0.001, 0.01, 10)
    for i, eps in enumerate(x):
        print(eps)
        dbscan = DBSCAN(eps=eps, metric="cosine")
        clusters = dbscan.fit_predict(matrix)
        num_clusters[i] = max(clusters)

    plt.plot(x, num_clusters)
    plt.show()

    dbscan = DBSCAN(eps=0.08, metric="cosine")
    clusters = dbscan.fit_predict(matrix)
    plt.scatter(X, Y, picker=True, c=clusters[:num_to_plot])

    def onpick(event):
        ind = event.ind
        if len(ind) > 1:
            i = int(ind[0])
            print(f"{len(ind)} images with the same tags.")
        else:
            i = int(ind)

        url = idToURL(file_ids[i])
        webbrowser.open(url)
        print(f"Cluster: {clusters[i]}")
        print(image_tags[i])
        print("\n")

    plt.connect("pick_event", onpick)
    plt.show()


def plot_tsne(matrix, file_ids, image_tags, num_to_plot=1000, cache_file="t_sne.cache"):
    load_cache = True
    if load_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            reduced = pickle.load(f)
    else:
        t_sne = TSNE(
            n_components=2,
            init="pca",
            n_iter=20000,
            learning_rate="auto",
            perplexity=10,
            metric="euclidean",
            verbose=1,
            n_jobs=-1,
        )
        reduced = t_sne.fit_transform(matrix[:num_to_plot, :])

        with open(cache_file, "wb") as f:
            pickle.dump(reduced, f)

    def onpick(event):
        ind = event.ind
        if len(ind) > 1:
            i = int(ind[0])
            print(f"{len(ind)} images with the same tags.")

            tag_counts = dict()
            for j in ind:
                for t in image_tags[j]:
                    try:
                        tag_counts[t] += 1
                    except KeyError:
                        tag_counts[t] = 1

            print(sorted(tag_counts.items(), key=lambda x: -x[1])[:10])

        else:
            i = int(ind)
            print(image_tags[i])

        url = idToURL(file_ids[i])
        # webbrowser.open(url)
        print("\n")

    # fig = plt.figure()
    # ax : plt.Axes = fig.add_subplot(projection='3d')

    X = reduced[:num_to_plot, 0]
    Y = reduced[:num_to_plot, 1]
    # Z = reduced[:num_to_plot, 2]

    clusters = dbscan(reduced[:num_to_plot], 100)
    print(max(clusters))
    plt.scatter(X, Y, picker=True, c=clusters)
    # plt.xlim([-1000, 1000])
    # plt.ylim([-1000, 1000])
    plt.connect("pick_event", onpick)
    plt.show()

    clusters_with_counts = count_tags_per_cluster(image_tags, clusters)

    return clusters_with_counts


def glove_to_clusters(client, metadata):
    glove_embeddings = load_glove("glove_embeddings30k.pt", "vocab30000.txt")
    embeddings, ids, tags = glove_images(client, glove_embeddings, metadata)
    clusters = plot_tsne(embeddings, ids, tags)
    os.mkdir("output")
    for c in clusters:
        for img in clusters[c]["images"]:
            file_id = metadata[img]["file_id"]
            r = client.get_thumbnail(file_id=file_id)
            image = Image.open(BytesIO(r.content))
            if not os.path.exists(f"output/{c}"):
                os.mkdir(f"output/{c}")
            image.save(f"output/{c}/{file_id}.{image.format}")


def count_tags_per_cluster(tags, clusters):

    cluster_tag_count = {}
    for i, (tag_set, cluster) in enumerate(zip(tags, clusters)):
        if cluster not in cluster_tag_count:
            cluster_tag_count[int(cluster)] = {"tags": {}, "images": []}

        cluster_tag_count[int(cluster)]["images"].append(i)
        for t in tag_set:
            try:
                cluster_tag_count[cluster]["tags"][t] += 1
            except KeyError:
                cluster_tag_count[cluster]["tags"][t] = 1

    for c in cluster_tag_count:
        tags = cluster_tag_count[c]["tags"]
        cluster_tag_count[c]["tags"] = sorted(tags.items(), key=lambda x: -x[1])

    return cluster_tag_count
