import itertools
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np
from networkx import graph_edit_distance
from sklearn.cluster import OPTICS


def clusterize_graphs(
    graphs: List[nx.Graph],
    fn: Callable[[nx.Graph, nx.Graph], float] = graph_edit_distance,
) -> List[int]:
    n = len(graphs)
    futures: List[Tuple[int, int, Future]] = []
    dist_mat = np.zeros(shape=(n, n))
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, j in itertools.product(range(n), range(n)):
            futures.append((i, j, executor.submit(fn, graphs[i], graphs[j])))
    for i, j, future in futures:
        dist_mat[i][j] = future.result()
    clustering = OPTICS(metric="precomputed", n_jobs=4).fit(dist_mat)
    return clustering.labels_
