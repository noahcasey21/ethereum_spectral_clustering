import numpy as np

def naive_transform(adj : np.array) -> np.array:
    return adj + adj.T

def bibliosymmetric_transform(adj : np.array) -> np.array:
    return adj @ adj.T + adj.T @ adj

def degree_discount_transform(adj : np.array) -> np.array:
    '''
    Adjacency matrix is of form
    row: outgoing
    column: incoming
    '''
    D_out = np.diag(np.sum(adj, axis=1))
    D_in = np.diag(np.sum(adj, axis=0))
    p = 0.5

    B = (D_out ** p) @ adj @ (D_in ** p) @ adj.T @ (D_out ** p)
    C = (D_in ** p) @ adj.T @ (D_out ** p) @ adj @ (D_in ** p)

    return B + C

def random_walk(adj : np.array) -> np.array:
    P, PI = get_markov_components(adj)

    return (np.diag(PI) @ P + P.T @ np.diag(PI)) / 2

def network_embedding(adj : np.array) -> np.array:
    '''
    Returns graph Laplacian using idea similar to random walk symmetrization 
    '''
    _, PI = get_markov_components(adj)

    return np.diag(PI) - random_walk(adj)

def get_markov_components(adj : np.array) -> tuple:
    P = adj  #pseudo transition matrix

    PI = np.ones(P.shape[0]) / P.shape[0]
    
    for _ in range(1000):
        PI_new = np.dot(PI, P)

        if np.allclose(PI, PI_new):
            break
        PI = PI_new

    return P, PI

def normalize_adj(adj : np.array) -> np.array:
    lower = np.min(adj)
    upper = np.max(adj)

    return (adj - lower) / (upper - lower)