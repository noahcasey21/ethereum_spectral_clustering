import numpy as np
from scipy.sparse import coo_matrix, save_npz
from os import listdir
import gzip
import pickle
from tqdm import tqdm
import random

def extract_adjacency(keep_nodes : int, random_seed=0) -> None:
    '''
    Create and save adjacency matrices from transaction data using nodes with highest transaction amounts

    keep_nodes: Number of nodes to randomly select to prevent disk overflow
    random_seed: random state for reprocity
    '''

    #read nodes
    with open('./data/preprocessing/nodes/nodes.pkl', 'rb') as f:
        nodes = pickle.load(f)
        f.close()

    nodes = nodes[: keep_nodes]

    path = './data/preprocessing/edges/'
    edgefiles = [path + f for f in listdir(path)]
    
    size = len(nodes)
    adj_mat = np.zeros((size, size))

    pbar = tqdm(total=len(edgefiles))
    for edgefile in edgefiles:
        pbar.update()
        with gzip.open(edgefile, 'rb') as f:
            edgelist = pickle.load(f)
            f.close()

        for sender, recipient, amount in edgelist:
            if sender in nodes and recipient in nodes:
                sender_idx = nodes.index(sender)
                
                sender_idx = nodes.index(sender)
                recipient_idx = nodes.index(recipient)         
                adj_mat[sender_idx, recipient_idx] += amount 

    pbar.close()

    print('Removing zeroed entries from nodes')
    # remove zeroed entries 
    remove_idx = []
    for i in range(adj_mat.shape[0]):
        if np.all(adj_mat[i, :] == 0) and np.all(adj_mat[:, i] == 0):
            remove_idx.append(i)

    cleaned_nodes = []
    for i in range(len(nodes)):
        if i not in remove_idx:
            cleaned_nodes.append(nodes[i])
        else:
            remove_idx.remove(i)

    print('Removing zeroed entries from adjacency')
    cleaned_adj_mat_tmp = np.delete(adj_mat, remove_idx, axis=0)
    cleaned_adj_mat = np.delete(cleaned_adj_mat_tmp, remove_idx, axis=1)

    print('Saving...')
    # save postprocessing data
    with open('./data/postprocessing/nodes.pkl', 'wb') as f:
        pickle.dump(cleaned_nodes, f)

    np.save('./data/postprocessing/adjacency.npy', cleaned_adj_mat)

    # convert to sparse matrix
    sparse_adj_mat = coo_matrix(cleaned_adj_mat)
    
    save_npz('./data/postprocessing/sparse_adjacency.npz', sparse_adj_mat)

if __name__ == '__main__':
    extract_adjacency(1000)