import numpy as np
from sklearn.decomposition import PCA

class tSNE:
    def __init__(self, n_dims=2, perplexity=20, lr=0.01, GD_iters=100):
        self.n_dimensions = n_dims
        self.perplexity = perplexity
        self.lr = lr
        self.GD_iters = GD_iters
        self.n_dims = n_dims

    def _find_sigma(self, distances_i: np.array):
        sigma_min = 1e-10
        sigma_max = 1000
        tol = 1e-5

        for _ in range(50):
            sigma = (sigma_min + sigma_max) / 2
            p = np.exp(-distances_i / (2 * sigma**2))
            p /= p.sum()

            entropy = -np.sum(p * np.log2(p + 1e-10))
            current_perplexity = 2**entropy

            # check convergence
            if abs(current_perplexity - self.perplexity) < tol:
                break
            
            # moving the boundaries of search
            if current_perplexity < self.perplexity:
                sigma_min = sigma
            else:
                sigma_max = sigma
        return sigma
    
    def _original_space_similarity(self, X): # X - original space. Returns matrix (n, n) with samples similarity
        distances = np.sum((X[:, None] - X) ** 2, axis=2) # distances between samples in original space
        res = np.zeros((len(X), len(X))) # similarity matrix (output)

        for i in range(len(X)):
            sigma_i = self._find_sigma(distances[i]) # find sigma for 1 sample
            res_i = np.exp(-distances[i] / (2 * sigma_i**2))
            res_i[i] = 0
            res_i /= res_i.sum()
            res[i] = res_i
        # symmetrize
        res = (res + res.T) / (2 * len(res))
        return res

    def _new_space_similarity(self, X):
        distances = np.sum((X[:, None] - X) ** 2, axis=2) # euclidian dist
        inv_dist = (1 + distances) ** (-1) # t-distribution
        np.fill_diagonal(inv_dist, 0) # self-similarity of sample = 0
        res = inv_dist / np.sum(inv_dist) # norm
        return res
    
    def _loss(self, original_similarity, new_similarity): # not used, but can be calculated in GD e.g. for a graph
        epsilon = 1e-10
        ratio = (original_similarity + epsilon) / (new_similarity + epsilon)
        loss = np.sum(original_similarity * np.log(ratio)) - np.sum(original_similarity) + np.sum(new_similarity)
        return np.round(loss, 3)
    
    def fit_transform(self, X, verbose=True):
        X = np.array(X)

        # initialization of a new space with PCA
        pca = PCA(n_components=self.n_dims)
        new_space = pca.fit_transform(X)
        orig_space_sim = self._original_space_similarity(X)

        # gradient descent
        for iter in range(self.GD_iters):
            new_space_sim = self._new_space_similarity(new_space)
            grad = np.zeros_like(new_space)

            # for each sample calculate a gradient
            for i in range(len(new_space)):
                diff = new_space[i] - new_space
                inv_dist = (1 + np.sum(diff ** 2, axis=1)) ** (-1)
                factor = (orig_space_sim[i] - new_space_sim[i]) * inv_dist
                grad[i] = 4 * np.dot(factor, diff)
            
            new_space -= self.lr * grad

            if verbose and iter % 100 == 0:
                loss = self._loss(orig_space_sim, new_space_sim)
                print(f"Iteration: {iter + 1}. Loss = {loss}")
        
        return new_space

