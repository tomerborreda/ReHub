import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import Linear as Linear_pyg

from graphgps.layer import rehub_utils
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE


class ReHub(nn.Module):
    """Global ReHub Module.
    
    This module holds the global components of the "Long-Range Spoke Update Layer":
    - Spokes to Hubs
    - Hubs to Hubs
    - Hubs to Spokes
    - Hub Reassignment 
    """

    def __init__(self, dim_h, num_heads, rehub_cfg, attn_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.rehub_cfg = rehub_cfg

        self.s_to_h_layer = GATv2Conv(dim_h, dim_h // num_heads, heads=num_heads, add_self_loops=False, dropout=attn_dropout)
        self.hubs_self_layer = GATv2Conv(dim_h, dim_h // num_heads, heads=num_heads, add_self_loops=False, dropout=attn_dropout)
        self.h_to_s_layer = GATv2Conv(dim_h, dim_h // num_heads, heads=num_heads, add_self_loops=False, dropout=attn_dropout)

    def forward(self, h_local, batch):
        h_global = batch.hub_features

        # Spokes to Hubs
        h_global = self.s_to_h_layer(x=(h_local, h_global), edge_index=batch.spokes_hubs_edge_index)

        # Hubs to Hubs
        hubs_hubs_edge_index = rehub_utils.fully_connected_edge_index(batch.hubs_batch)
        h_global = self.hubs_self_layer(x=h_global, edge_index=hubs_hubs_edge_index)

        # Hubs to Spokes
        hubs_spokes_edge_index = torch.stack([batch.spokes_hubs_edge_index[1], batch.spokes_hubs_edge_index[0]], dim=0)
        h_local, spokes_attention_weights = self.h_to_s_layer(x=(h_global, h_local), edge_index=hubs_spokes_edge_index, return_attention_weights=True)

        if self.rehub_cfg.logging.plot_metrics:
            batch.log_utilization.append(rehub_utils.get_graph_hub_utilization(batch.spokes_hubs_edge_index, batch.hubs_batch, batch.num_hubs))
            batch.log_bhattacharyya.append(rehub_utils.get_graph_bhattacharyya(batch.spokes_hubs_edge_index, batch.hubs_batch, batch.num_hubs))

        # Reassign spokes to hubs. I.e. update the assignment edge_index.
        if self.rehub_cfg.reassignment_strategy is None:
            pass
        elif self.rehub_cfg.reassignment_strategy == 'k_closest_by_attention':
            batch.spokes_hubs_edge_index = rehub_utils.assign_edges_to_k_closest(h_global, batch.hubs_batch, batch.num_hubs, batch.batch, rehub_utils.get_closest_hub_from_spokes_attention_weights(spokes_attention_weights, batch.num_hubs), batch.hubs_per_spoke)
        elif self.rehub_cfg.reassignment_strategy == 'k_closest_by_distance':
            batch.spokes_hubs_edge_index = rehub_utils.assign_edges_to_k_closest(h_global, batch.hubs_batch, batch.num_hubs, batch.batch, rehub_utils.get_closest_hub_from_spokes_emb(h_local, h_global, batch.spokes_hubs_edge_index, batch.num_hubs), batch.hubs_per_spoke)
        else:
            raise ValueError(f"Unsupported reassignment strategy: {self.rehub_cfg.reassignment_strategy}")

        batch.hub_features = h_global
        return h_local


class ReHubLayer(nn.Module):
    """Local MPNN + ReHub layer, or as depicted in the paper "Long-Range Spoke Update Layer"""

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False, rehub_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == "GCN2":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCN2Conv(dim_h, 0.3)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'ReHub':
            self.self_attn = ReHub(dim_h, num_heads, rehub_cfg, attn_dropout=self.attn_dropout)
        else:
            raise ValueError(f"Unsupported global model: {global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr)
                else:
                    if self.local_gnn_type == 'GCN2':
                        h_local = self.local_model(h, h, batch.edge_index)
                    else:
                        h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type == 'ReHub':
                h_attn = self.self_attn(h_local, batch)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
