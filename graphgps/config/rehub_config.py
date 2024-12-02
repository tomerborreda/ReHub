from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_rehub')
def set_cfg_gt(cfg):
    """Configuration for ReHub model."""

    cfg.rehub = CN()
    cfg.rehub.prep = False
    cfg.rehub.hubs_per_spoke = 3
    cfg.rehub.num_hubs_type = "D"
    cfg.rehub.num_hubs = 1.0  # For the dynamic case this is a ratio from the number of nodes. For the static case this is the number of hubs and would be rounded using int()
    cfg.rehub.spokes_mlp_before_hub_agg = True
    cfg.rehub.reassignment_strategy = "k_closest_by_attention"
    cfg.rehub.learnable_hubs = False

    cfg.rehub.logging = CN()
    cfg.rehub.logging.plot_metrics = False
    cfg.rehub.logging.log_cuda_time_and_gpu_memory = False
    cfg.rehub.logging.profile_memory = False  # Cautious! Captures a memory snapshot and get overflow storage
