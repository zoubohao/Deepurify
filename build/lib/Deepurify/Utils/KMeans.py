import numpy as np
import scipy


def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in range(n):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise ValueError('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods


def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()


def weighted_farthest_first_traversal(points, weights, k):
    traversed = []

    # Choose the first point randomly (weighted)
    i = np.random.choice(len(points), size=1, p=weights)[0]
    traversed.append(i)

    # Find remaining n - 1 maximally separated points
    for _ in range(k - 1):
        max_dst, max_dst_index = 0, None

        for i in range(len(points)):
            if i not in traversed:
                dst = dist(i, traversed, points)
                weighted_dst = weights[i] * dst

                if weighted_dst > max_dst:
                    max_dst = weighted_dst
                    max_dst_index = i

        traversed.append(max_dst_index)

    return traversed


class COPKMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X, y=None, ml=[], cl=[]):
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize cluster centers
        cluster_centers = self._init_cluster_centers(X)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            # print(iteration)
            prev_cluster_centers = cluster_centers.copy()

            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, self._dist, ml_graph, cl_graph)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-4, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _init_cluster_centers(self, X):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]

    def _dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def _assign_clusters(self, *args):
        max_retries_cnt = 1000

        for retries_cnt in range(max_retries_cnt):
            try:
                return self._try_assign_clusters(*args)
            except:
                continue

        raise ValueError("ClusteringNotFoundException")

    def _try_assign_clusters(self, X, cluster_centers, dist, ml_graph, cl_graph):
        labels = np.full(X.shape[0], fill_value=-1)

        data_indices = list(range(X.shape[0]))
        np.random.shuffle(data_indices)

        for i in data_indices:
            distances = np.array([dist(X[i], c) for c in cluster_centers])

            for cluster_index in distances.argsort():
                if not self._violates_constraints(i, cluster_index, labels, ml_graph, cl_graph):
                    labels[i] = cluster_index
                    break

            if labels[i] < 0:
                raise ValueError("ClusteringNotFoundException")

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise ValueError("EmptyClustersException")

        return labels

    def _violates_constraints(self, i, cluster_index, labels, ml_graph, cl_graph):
        for j in ml_graph[i]:
            if labels[j] > 0 and cluster_index != labels[j]:
                return True

        for j in cl_graph[i]:
            if cluster_index == labels[j]:
                return True

        return False

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
    
    