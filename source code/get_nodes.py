import os
import gzip
import pickle
from tqdm import tqdm
from collections import defaultdict

'''
For early interrupt of data extract, gets nodes
'''

def get_nodes():
    '''
    Gets nodes from edges list
    Ranks them by number of transactions they appear in
    '''
    path = './data/preprocessing/edges/'
    edgefiles = [path + f for f in os.listdir(path)]

    nodes = defaultdict(int)
    
    pbar = tqdm(total=len(edgefiles))
    for edgefile in edgefiles:
        pbar.update()
        with gzip.open(edgefile, 'rb') as f:
            edgelist = pickle.load(f)
            f.close()

        for sender, recipient, _ in edgelist:
            nodes[sender] += 1
            nodes[recipient] += 1

    node_sort = dict(sorted(nodes.items(), key=lambda item: item[1], reverse=True))
    sorted_nodes = list(node_sort.keys())

    pbar.close()
    with open('./data/preprocessing/nodes/nodes.pkl', 'wb') as f:
        pickle.dump(list(sorted_nodes), f)

if __name__ == '__main__':
    get_nodes()