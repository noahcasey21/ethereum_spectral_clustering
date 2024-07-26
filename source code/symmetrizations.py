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

    return (PI @ P + P.T @ PI) / 2

def network_embedding(adj : np.array) -> np.array:
    '''
    Returns graph Laplacian using idea similar to random walk symmetrization 
    '''
    _, PI = get_markov_components(adj)

    return PI - random_walk(adj)

def get_markov_components(adj : np.array) -> tuple:
    P = adj / np.sum(adj, axis=0) #transition matrix
    
    A = np.transpose(P) - np.eye(P.shape[0])
    A = np.vstack([A, np.ones(P.shape[0])])

    b = np.zeros(P.shape[0])
    b = np.append(b, 1)

    PI = np.linalg.lstsq(A, b, rcond=None)[0]

    return P, PI

def normalize_adj(adj : np.array) -> np.array:
    lower = np.min(adj)
    upper = np.max(adj)

    return (adj - lower) / (upper - lower)