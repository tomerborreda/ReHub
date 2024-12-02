import torch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_max, scatter_min


def unbached_hub_idx_to_bached(unbatched_hub_idx, spoke_batch, num_hubs_per_batch):
    num_hubs_cum_sizes = torch.cumsum(num_hubs_per_batch, dim=0)
    num_hubs_cum_sizes = torch.cat((torch.tensor([0], device=spoke_batch.device), num_hubs_cum_sizes))
    return num_hubs_cum_sizes[spoke_batch] + unbatched_hub_idx


def bached_hub_idx_to_unbached(batched_hub_idx, num_hubs_per_batch):
    batched_to_unbatched_idx = torch.tensor([hub_idx for hubs_count in num_hubs_per_batch for hub_idx in range(hubs_count)], device=batched_hub_idx.device)
    return batched_to_unbatched_idx[batched_hub_idx]


def assign_edges(K_best, spoke_batch, batch_num_hubs):
    assert len(K_best.shape) == 2

    edge_index = torch.zeros(2, spoke_batch.size(0), dtype=torch.int64, device=spoke_batch.device)
    edge_index[0] = torch.arange(spoke_batch.size(0), device=spoke_batch.device)
    edge_index[1] = K_best[:,0]
    
    for i in range(1, K_best.size(1)):
        # For each spoke use K only up to the number of hubs in that graph
        # Find the indices of the spokes to add another hub to
        spokes_bool = batch_num_hubs[spoke_batch] > i

        # Only add a new hub to the spokes that have more hubs to add to
        spokes_idx = torch.arange(spokes_bool.size(0), device=spoke_batch.device)[spokes_bool]
        hubs_idx = K_best[spokes_bool, i]
        
        new_edge_index = torch.stack((spokes_idx, hubs_idx), dim=0)
        edge_index = torch.cat((edge_index, new_edge_index), dim=1)

    return edge_index


def assign_edges_to_k_closest(hub_features, hub_batch, batch_num_hubs, spoke_batch, spoke_closest_hub_idx_unbached, hubs_per_spoke):
    hub_features, hub_mask = to_dense_batch(hub_features, hub_batch)

    # Calculate the distance between all hubs
    diff = hub_features.unsqueeze(1) - hub_features.unsqueeze(2)
    hub_distances = torch.norm(diff, dim=-1)

    hub_mask_sym = hub_mask.unsqueeze(2).repeat(1,1,hub_mask.size(1))
    hub_mask_sym.transpose(1,2)[hub_mask_sym == False] = False
    hub_distances[hub_mask_sym == False] = 1e16

    # Find the indices of the K closest hubs (including the original one)
    _, top_k_indices = torch.topk(hub_distances, hubs_per_spoke, largest=False, dim=-1)

    # For each node choose the closet K-1 hubs which are closest to the current one
    K_best = top_k_indices[spoke_batch, spoke_closest_hub_idx_unbached, :]

    for i in range(0, hubs_per_spoke):
        K_best[:,i] = unbached_hub_idx_to_bached(K_best[:,i], spoke_batch, batch_num_hubs)

    return assign_edges(K_best, spoke_batch, batch_num_hubs)


def get_closest_hub_from_spokes_attention_weights(spokes_attention_weights, num_hubs_per_batch):
    closest_hubs_bached = spokes_attention_weights[0][0][scatter_max(spokes_attention_weights[1][:,0], spokes_attention_weights[0][1])[1]]
    return bached_hub_idx_to_unbached(closest_hubs_bached, num_hubs_per_batch)


def get_closest_hub_from_spokes_emb(spokes_emb, hubs_emb, spokes_hubs_edge_index, num_hubs_per_batch):
    spokes_emb = spokes_emb[spokes_hubs_edge_index[0]]
    hubs_emb = hubs_emb[spokes_hubs_edge_index[1]]
    diff = spokes_emb - hubs_emb
    distances = torch.norm(diff, dim=-1)

    closest_hub_idx_per_spoke = scatter_min(distances, spokes_hubs_edge_index[0])[1]
    closest_hubs_bached = spokes_hubs_edge_index[1][closest_hub_idx_per_spoke]
    return bached_hub_idx_to_unbached(closest_hubs_bached, num_hubs_per_batch)


def get_furthest_hub_from_spokes_emb(spokes_emb, hubs_emb, spokes_hubs_edge_index):
    spokes_emb = spokes_emb[spokes_hubs_edge_index[0]]
    hubs_emb = hubs_emb[spokes_hubs_edge_index[1]]
    diff = spokes_emb - hubs_emb
    distances = torch.norm(diff, dim=-1)

    furthest_hub_idx_per_spoke = scatter_max(distances, spokes_hubs_edge_index[0])[1]
    return furthest_hub_idx_per_spoke


def fully_connected_edge_index(hubs_batch):
    # Find unique batches and inverse indices
    _, inverse_indices = torch.unique(hubs_batch, return_inverse=True)
    
    # Create a meshgrid of indices
    indices = torch.arange(hubs_batch.size(0), device=hubs_batch.device)
    sources, destinations = torch.meshgrid(indices, indices, indexing='ij')
    
    # Mask to keep only connections within the same batch
    mask = inverse_indices[sources] == inverse_indices[destinations]
    
    # Apply the mask and flatten the source and destination indices
    sources = sources[mask].flatten()
    destinations = destinations[mask].flatten()
    
    # Stack sources and destinations to create the final tensor
    connection_tensor = torch.stack((sources, destinations))

    return connection_tensor


### For Metrics ###


def get_graph_hub_utilization(spokes_hubs_edge_index, hub_batch, num_hubs_per_graph):
    """Returns the precentage of hubs that are connected to a spoke"""
    hub_idx = spokes_hubs_edge_index[1]  # Hubs indices spokes are connected to
    unique_hub_idx = torch.unique(hub_idx)
    unique_hub_batch = hub_batch[unique_hub_idx]
    num_hubs_with_node_connection = torch.bincount(unique_hub_batch, minlength=hub_batch.max().item() + 1)  # For each graph we count the number of hubs that are connected to a spoke
    return (num_hubs_per_graph - num_hubs_with_node_connection)/num_hubs_per_graph


def get_graph_bhattacharyya(spokes_hubs_edge_index, hub_batch, num_hubs_per_graph):
    nodes_per_hub = torch.unique(spokes_hubs_edge_index[1], return_counts=True)[1]
    graph_idx_for_each_value = hub_batch[torch.unique(spokes_hubs_edge_index[1])]

    sum_nodes_per_hub = scatter_add(nodes_per_hub, graph_idx_for_each_value)
    nodes_per_hub_normalized = nodes_per_hub / sum_nodes_per_hub[graph_idx_for_each_value]

    bhattacharyya_per_graph = nodes_per_hub_normalized * (1/num_hubs_per_graph[graph_idx_for_each_value])
    bhattacharyya_per_graph = torch.sqrt(bhattacharyya_per_graph)
    bhattacharyya_per_graph = scatter_add(bhattacharyya_per_graph, graph_idx_for_each_value)
    return bhattacharyya_per_graph
