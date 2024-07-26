import requests
import ast
from tqdm import tqdm
import gzip
import pickle

def pull_block_data(startblock : str, endblock : str):
    '''
    Pull Ethereum transaction block data

    startblock: hexidecimal string for starting block
    endblock: hexidecimal string for ending block
    '''

    #start and end block
    startblock_int = int(startblock, 16)
    endblock_int = int(endblock, 16)

    #https url
    url = "https://base-mainnet.core.chainstack.com/2fc1de7f08c0465f6a28e3c355e0cb14/"

    #headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    successful_requests = 0

    nodes = set()

    pbar = tqdm(total=(endblock_int - startblock_int))
    i = startblock_int
    while i <= endblock_int:
        #initialize weighted directional graph
        edges = [] #list of tuples [(sender, recipient, amount), ...]
        pbar.update()

        #body
        payload = {
            "jsonrpc": "2.0",
            "method": "trace_block",
            "id": 1,
            "params": [hex(i)]
        }

        # get block data
        try:
            response = requests.post(url, json=payload, headers=headers)
            data = ast.literal_eval(response.text)['result']
        except:
            i += 1
            continue 

        successful_requests += 1

        for dat in data:
            try:
                #try to get data
                sender = dat['action']['from']
                recipient = dat['action']['to']
                amount = int(dat['action']['value'], 16)

                #add edge from sender to recipient with weight amount
                if amount != 0 and sender != recipient:
                    nodes.add(sender)
                    nodes.add(recipient)
                    edges.append((sender, recipient, amount))

            except KeyError:
                continue

        #save graph to compressed file
        file_name = 'edges_block_' + str(i)
        with gzip.open(f'./data/preprocessing/edges/{file_name}.pkl.gzip', 'wb') as f:
            pickle.dump(edges, f)
        i += 1

    with open('./data/preprocessing/nodes/nodes.pkl', 'wb') as f:
        pickle.dump(list(nodes), f)

    pbar.close()
    print(f'Block requests made: {endblock_int - startblock_int}\nBlock requests successful: {successful_requests}')


#driver code
if __name__ == "__main__":
    start_block_int = 16_000_000
    start_block = hex(start_block_int)
    end_block_int = 17_000_000
    end_block = hex(end_block_int)

    pull_block_data(start_block, end_block)