#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    wandb_project=$1
    dataset=$2
    cfg_suffix=$3
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$4

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}-${cfg_suffix}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides} wandb.project ${wandb_project}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    for i in {1..5}; do
        SEED=$RANDOM
        script="sbatch ${slurm_directive} -J ${cfg_suffix}-${dataset} run_experiments/wrapper.sb ${main} --repeat 1 seed ${SEED} ${common_params}"
        echo $script
        eval $script
    done
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done


cfg_dir="configs/ReHub"
slurm_directive="--mem=0 --gres=gpu:1 --ntasks-per-node=1 --nodes=1 -o .slurm_logs/out_job%j.txt -e .slurm_logs/err_job%j.txt --cpus-per-task=10"

DATASET="peptides-func"
WANDB_PROJECT="ReHub-ablations-peptides-func"
# Adding parts
# GNN Only
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.GNN.peptides-func.CustomGatedGCN.5run model.type GPSModel gt.layer_type CustomGatedGCN+None rehub.prep False"
# Learnable Hubs, Const
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.LearnableHubs.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 12.0 rehub.num_hubs_type S rehub.prep False rehub.learnable_hubs True rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg False"
# Hubs initialized from spokes, Const, cluster calculated by metis, No Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.HubsMetisConst.NoSpokeEnc.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 12.0 rehub.num_hubs_type S rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg False"
# Hubs initialized from spokes, Dynamic, cluster calculated by metis, No Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.HubsMetisDynamic.NoSpokeEnc.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg False"
# Hubs initialized from spokes, Dynamic, cluster calculated by metis, With Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.HubsMetisDynamic.NoSpokeEnc.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg True"
# Reassignment, No Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.Reassignment.NoSpokeEnc.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy k_closest_by_attention rehub.spokes_mlp_before_hub_agg False"
# Reassignment
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.Reassignment.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy k_closest_by_attention rehub.spokes_mlp_before_hub_agg True"


# Varying ratio of hubs with K = 5
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.0.5.K5.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 0.5 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.1.K5.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.2.K5.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 2.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.3.K5.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 3.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.4.K5.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 4.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.5.K5.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 5.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"


# Varying of ratio of hubs with K = 3
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.0.5.K3.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 0.5 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.1.K3.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.2.K3.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 2.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.3.K3.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 3.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.4.K3.peptidse-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 4.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.5.K3.peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 5.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"



DATASET="vocsuperpixels"
WANDB_PROJECT="ReHub-ablations-vocsuperpixels"
# Adding parts
# GNN Only
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.GNN.vocsuperpixels.CustomGatedGCN.5run model.type GPSModel gt.layer_type CustomGatedGCN+None rehub.prep False"
# Learnable Hubs, Const
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.LearnableHubs.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 22.0 rehub.num_hubs_type S rehub.prep False rehub.learnable_hubs True rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg False"
# Hubs initialized from spokes, Const, cluster calculated by metis, No Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.HubsMetisConst.NoSpokeEnc.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 22.0 rehub.num_hubs_type S rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg False"
# Hubs initialized from spokes, Dynamic, cluster calculated by metis, No Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.HubsMetisDynamic.NoSpokeEnc.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg False"
# Hubs initialized from spokes, Dynamic, cluster calculated by metis, With Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.HubsMetisDynamic.NoSpokeEnc.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy None rehub.spokes_mlp_before_hub_agg True"
# Reassignment, No Spoke Encoder
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.Reassignment.NoSpokeEnc.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy k_closest_by_attention rehub.spokes_mlp_before_hub_agg False"
# Reassignment
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag PartByPart.Reassignment.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.reassignment_strategy k_closest_by_attention rehub.spokes_mlp_before_hub_agg True"


# Varying ratio of hubs with K = 5
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.0.5.K5.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 0.5 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.1.K5.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.2.K5.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 2.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.3.K5.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 3.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.4.K5.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 4.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.5.K5.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 5.0 rehub.num_hubs_type D rehub.hubs_per_spoke 5"


# Varying ratio of hubs with K = 3
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.0.5.K3.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 0.5 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.1.K3.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 1.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.2.K3.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 2.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.3.K3.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 3.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.4.K3.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 4.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag HubsRatio.5.K3.vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.num_hubs 5.0 rehub.num_hubs_type D rehub.hubs_per_spoke 3"

