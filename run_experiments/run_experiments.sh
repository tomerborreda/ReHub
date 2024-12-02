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

WANDB_PROJECT="ReHub-Experiments"


DATASET="peptides-func"
# Sparse
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.CustomGatedGCN+LapPE+RWSE-ReHub.5run gt.layer_type CustomGatedGCN+ReHub posenc_RWSE.enable True dataset.node_encoder_name Atom+LapPE+RWSE"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.GCN-ReHub.5run gt.layer_type GCN+ReHub gt.dim_hidden 155"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.GINE-ReHub.5run gt.layer_type GINE+ReHub gt.n_heads 2"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.GCN2-ReHub.5run gt.layer_type GCN2+ReHub"

# Fully Connected (Use a large number of hubs_per_spoke (aka 50K) for max connecetivity)
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.FC.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.FC.CustomGatedGCN+LapPE+RWSE-ReHub.5run gt.layer_type CustomGatedGCN+ReHub posenc_RWSE.enable True dataset.node_encoder_name Atom+LapPE+RWSE rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.FC.GCN-ReHub.5run gt.layer_type GCN+ReHub gt.dim_hidden 155 rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.FC.GINE-ReHub.5run gt.layer_type GINE+ReHub gt.n_heads 2 rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-func.FC.GCN2-ReHub.5run gt.layer_type GCN2+ReHub rehub.hubs_per_spoke 50000"


DATASET="peptides-struct"
# Sparse
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.CustomGatedGCN+LapPE+RWSE-ReHub.5run gt.layer_type CustomGatedGCN+ReHub posenc_RWSE.enable True dataset.node_encoder_name Atom+LapPE+RWSE"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.GCN-ReHub.5run gt.layer_type GCN+ReHub gt.dim_hidden 155"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.GINE-ReHub.5run gt.layer_type GINE+ReHub gt.n_heads 2"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.GCN2-ReHub.5run gt.layer_type GCN2+ReHub"

# Fully Connected (Use a large number of hubs_per_spoke (aka 50K) for max connecetivity)
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.FC.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub  rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.FC.CustomGatedGCN+LapPE+RWSE-ReHub.5run gt.layer_type CustomGatedGCN+ReHub posenc_RWSE.enable True dataset.node_encoder_name Atom+LapPE+RWSE rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.FC.GCN-ReHub.5run gt.layer_type GCN+ReHub gt.dim_hidden 155 rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.FC.GINE-ReHub.5run gt.layer_type GINE+ReHub gt.n_heads 2 rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag peptides-struct.FC.GCN2-ReHub.5run gt.layer_type GCN2+ReHub rehub.hubs_per_spoke 50000"


DATASET="pcqm-contact"
# Sparse
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.CustomGatedGCN+LapPE+RWSE-ReHub.5run gt.layer_type CustomGatedGCN+ReHub posenc_RWSE.enable True dataset.node_encoder_name Atom+LapPE+RWSE"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.GCN-ReHub.5run gt.layer_type GCN+ReHub gt.dim_hidden 300 gt.layers 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.GINE-ReHub.5run gt.layer_type GINE+ReHub gt.dim_hidden 100 gt.layers 5"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.GCN2-ReHub.5run gt.layer_type GCN2+ReHub gt.dim_hidden 100 gt.n_heads 2 gt.layers 5"

# Fully Connected (Use a large number of hubs_per_spoke (aka 50K) for max connecetivity)
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.FC.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.FC.CustomGatedGCN+LapPE+RWSE-ReHub.5run gt.layer_type CustomGatedGCN+ReHub posenc_RWSE.enable True dataset.node_encoder_name Atom+LapPE+RWSE rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.FC.GCN-ReHub.5run gt.layer_type GCN+ReHub gt.dim_hidden 300 gt.layers 5 rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.FC.GINE-ReHub.5run gt.layer_type GINE+ReHub gt.dim_hidden 100 gt.layers 5 rehub.hubs_per_spoke 50000"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag pcqm-contact.FC.GCN2-ReHub.5run gt.layer_type GCN2+ReHub gt.dim_hidden 100 gt.n_heads 2 gt.layers 5 rehub.hubs_per_spoke 50000"


DATASET="vocsuperpixels"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag vocsuperpixels.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag vocsuperpixels.FC.CustomGatedGCN-ReHub.5run gt.layer_type CustomGatedGCN+ReHub rehub.hubs_per_spoke 50000"


DATASET="ogbn-arxiv"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag ogbn-arxiv.GCN-ReHub.5run rehub.logging.log_cuda_time_and_gpu_memory True"


DATASET="physics"
run_repeats ${WANDB_PROJECT} ${DATASET} ReHub "name_tag physics.GCN-ReHub.5run rehub.logging.log_cuda_time_and_gpu_memory True"
