import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (BatchNorm1dNode,
                                                   new_layer_config)
from torch_geometric.graphgym.register import register_network
from torch_scatter import scatter_mean

from graphgps.layer.rehub_layer import ReHubLayer
from graphgps.layer.rehub_utils import (assign_edges_to_k_closest,
                                        unbached_hub_idx_to_bached)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out, mult=2, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
    

def aggregate_spokes_to_hubs(batch, ff=None):
    """Aggregates spoke features to hub features."""
    x = ff(batch.x) if ff is not None else batch.x
    unique_indices = unbached_hub_idx_to_bached(batch.spoke_init_hub_idx, batch.batch, batch.num_hubs)
    batch.hub_features = scatter_mean(x, unique_indices, dim=0, dim_size=sum(batch.num_hubs))
    batch.hubs_batch = torch.tensor([batch_idx for batch_idx, hubs_count in enumerate(batch.num_hubs) for _ in range(hubs_count)]).to(batch.x.device)


class HubCreator(torch.nn.Module):
    """Hub Initialization Module

    This module initializes hubs according to the number of hub-cluster found in batch.spoke_to_hub_index.
    If the hubs are learnable then only a static amout of hubs is supported.
    """

    def __init__(self, dim_hidden, dropout, rehub_cfg):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.rehub_cfg = rehub_cfg
        if rehub_cfg.spokes_mlp_before_hub_agg:
            self.ff = FeedForward(dim_hidden, dim_hidden, dropout)
        else:
            self.ff = None

        if rehub_cfg.learnable_hubs:
            if rehub_cfg.num_hubs_type != "S":
                raise ValueError(f"Only static number of hubs is supported for learnable hubs but rehub_cfg.num_hubs_type = {rehub_cfg.num_hubs_type}")
            self.hub_features = nn.Parameter(torch.Tensor(int(rehub_cfg.num_hubs), dim_hidden))
            torch.nn.init.xavier_uniform_(self.hub_features)
            
    def forward(self, batch):
        if cfg.rehub.learnable_hubs:
            batch.hub_features = self.hub_features.repeat(batch.num_graphs, 1).to(batch.x.device)
            batch.hubs_batch = torch.arange(batch.num_graphs).repeat_interleave(int(cfg.rehub.num_hubs)).to(batch.x.device)
            batch.num_hubs = torch.full((batch.num_graphs,), int(cfg.rehub.num_hubs), dtype=torch.long).to(batch.x.device)
            batch.hubs_per_spoke = min(self.rehub_cfg.hubs_per_spoke, batch.num_hubs.max().item())

            spokes_idx = torch.arange(batch.batch.size(0), device=batch.batch.device).repeat_interleave(batch.hubs_per_spoke)
            hubs_idx = torch.randint(0, int(cfg.rehub.num_hubs), (batch.hubs_per_spoke * batch.batch.size(0),), device=batch.x.device)
            batch.spokes_hubs_edge_index = torch.stack([spokes_idx, hubs_idx], dim=0)
        else:
            if not hasattr(batch, 'spoke_init_hub_idx'):
                raise ValueError("batch does not have the spoke_init_hub_idx attribute. Enable rehub.prep.")
            aggregate_spokes_to_hubs(batch, ff=self.ff)
            batch.hubs_per_spoke = min(self.rehub_cfg.hubs_per_spoke, batch.num_hubs.max().item())
            batch.spokes_hubs_edge_index = assign_edges_to_k_closest(batch.hub_features, batch.hubs_batch, batch.num_hubs, batch.batch, batch.spoke_init_hub_idx, batch.hubs_per_spoke)

        if cfg.rehub.logging.plot_metrics:
            batch.log_utilization = []
            batch.log_bhattacharyya = []

        return batch

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

        self.hubs_creator = HubCreator(cfg.gt.dim_hidden, cfg.gt.dropout, cfg.rehub)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('ReHubModel')
class ReHubModel(torch.nn.Module):
    """ReHub Model Module"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(ReHubLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
                rehub_cfg=cfg.rehub,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
