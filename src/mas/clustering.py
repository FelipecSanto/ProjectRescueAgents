import os
import numpy as np
from sklearn.cluster import KMeans


def cluster_victims(victims: dict, n_clusters: int):
    """
    Agrupa vítimas em clusters com base em suas coordenadas (x, y).

    :param victims: dicionário no formato {id: ((x, y), sinais_vitais)}
    :param n_clusters: número de clusters (igual ao número de socorristas)
    :return: lista de clusters, cada um é uma lista de ids de vítimas
    """
    if len(victims) < n_clusters:
        raise ValueError(f"Número de vítimas ({len(victims)}) é menor que o número de clusters ({n_clusters})")

    victim_ids = list(victims.keys())
    coords = np.array([victims[vid][0] for vid in victim_ids])  # extrai (x,y) de cada vítima

    # Aplica KMeans para gerar os clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(coords)

    # Agrupa os ids por rótulo de cluster
    clusters = {i: [] for i in range(n_clusters)}
    for vid, label in zip(victim_ids, labels):
        clusters[label].append(vid)

    return list(clusters.values())


def save_clusters(clusters: list, victims: dict, output_dir: str = "clusters"):
    """
    Salva os clusters em arquivos cluster1.txt, cluster2.txt... no formato CSV.

    :param clusters: lista de clusters, cada um sendo uma lista de ids
    :param victims: dicionário com dados das vítimas no formato {id: ((x,y), sinais, classe?, gravidade?)}
    :param output_dir: diretório onde salvar os arquivos
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, cluster in enumerate(clusters, start=1):
        file_path = os.path.join(output_dir, f"cluster{i}.txt")
        with open(file_path, "w") as f:
            for vid in cluster:
                x, y = victims[vid][0]
                # gravidade e classe: se estiverem presentes, inclui; senão, usa vazio
                grav = ""
                classe = ""
                if len(victims[vid]) >= 4:
                    classe = victims[vid][2]
                    grav = victims[vid][3]
                f.write(f"{vid},{x},{y},{grav},{classe}\n")
