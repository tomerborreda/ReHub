import math

import numpy as np
import pymetis
import torch


def convert_edge_index_to_adjacency_list(edge_index, num_nodes):
    """Converts edge index to undirected adjacency list."""
    adjacency_list = [set() for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)

    adjacency_list = [list(e) for e in adjacency_list]
    return adjacency_list


def spoke_hub_clusters(data, num_hubs, num_hubs_type):
    """Partitions the graph into num_hubs clusters using the METIS library.
    
    Returns a tensor mapping each spoke to a hub index.
    """
    # First decide on the number of hubs for the graph
    num_nodes = data.x.shape[0]
    if num_hubs_type == 'D': # Dynamic
        if num_hubs is None:
            num_hubs = int(math.sqrt(num_nodes))
        else:
            num_hubs = int(num_hubs * math.sqrt(num_nodes))
            num_hubs = int(max(2, num_hubs))
    elif num_hubs_type == 'S': # Static
        num_hubs = int(num_hubs) # Keep num_hubs as is
    else:
        raise ValueError(f"num_hubs_type must be either 'D' or 'S'. Got {num_hubs_type}")
    
    # Then, partition the graph into num_hubs clusters
    adjacency_list = convert_edge_index_to_adjacency_list(data.edge_index, num_nodes)

    n_cuts, membership = pymetis.part_graph(num_hubs, adjacency=adjacency_list)

    node_to_hub = {}
    for hub_idx in range(num_hubs):
        nodes_idx = np.argwhere(np.array(membership) == hub_idx).ravel()
        for node_idx in nodes_idx:
            # Verify there are no duplicate nodes
            assert node_idx not in node_to_hub
            node_to_hub[node_idx] = hub_idx
    
    assert node_to_hub.keys() == set(range(num_nodes))

    spoke_to_hub_index_l = []
    for node_idx, hub_idx in sorted(node_to_hub.items(), key=lambda x: x[0]):
        spoke_to_hub_index_l.append(torch.tensor(hub_idx, dtype=torch.int64))

    # tensor of length number of spokes. For each spoke we have a hub index
    data.spoke_init_hub_idx = torch.stack(spoke_to_hub_index_l, dim=0)
    data.num_hubs = torch.tensor(num_hubs, dtype=torch.int64)
    return data
