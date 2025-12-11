#!/bin/bash

# MOM2 Samples Ablation Study
# Tests mom2_n_samples values: 10, 100, 1000, 10000
# Models: Qwen2.5-7B-Instruct, Llama3-8B-Instruct

set -e  # Exit on error

cd /home/pg2860/anyedit/AnyEdit

# Create results directory for this ablation
RESULTS_DIR="/home/pg2860/anyedit/AnyEdit/output/mom2_ablation_results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Starting MOM2 Samples Ablation Study"
echo "Results will be saved to: $RESULTS_DIR"
echo "=========================================="

# Qwen2.5-7B-Instruct with mom2_n_samples=10
echo ""
echo "[1/8] Qwen2.5-7B-Instruct with mom2_n_samples=10"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=mom2_samples_ablation/Qwen2.5-7B-Instruct_mom2_10.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_10_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_10_summary.txt"

# Qwen2.5-7B-Instruct with mom2_n_samples=100
echo ""
echo "[2/8] Qwen2.5-7B-Instruct with mom2_n_samples=100"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=mom2_samples_ablation/Qwen2.5-7B-Instruct_mom2_100.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_100_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_100_summary.txt"

# Qwen2.5-7B-Instruct with mom2_n_samples=1000
echo ""
echo "[3/8] Qwen2.5-7B-Instruct with mom2_n_samples=1000"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=mom2_samples_ablation/Qwen2.5-7B-Instruct_mom2_1000.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_1000_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_1000_summary.txt"

# Qwen2.5-7B-Instruct with mom2_n_samples=10000
echo ""
echo "[4/8] Qwen2.5-7B-Instruct with mom2_n_samples=10000"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=mom2_samples_ablation/Qwen2.5-7B-Instruct_mom2_10000.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_10000_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Qwen2.5-7B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Qwen2.5-7B-Instruct_mom2_10000_summary.txt"

# Llama3-8B-Instruct with mom2_n_samples=10
echo ""
echo "[5/8] Llama3-8B-Instruct with mom2_n_samples=10"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=mom2_samples_ablation/Llama3-8B-Instruct_mom2_10.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json "$RESULTS_DIR/Llama3-8B-Instruct_mom2_10_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Llama3-8B-Instruct_mom2_10_summary.txt"

# Llama3-8B-Instruct with mom2_n_samples=100
echo ""
echo "[6/8] Llama3-8B-Instruct with mom2_n_samples=100"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=mom2_samples_ablation/Llama3-8B-Instruct_mom2_100.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json "$RESULTS_DIR/Llama3-8B-Instruct_mom2_100_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Llama3-8B-Instruct_mom2_100_summary.txt"

# Llama3-8B-Instruct with mom2_n_samples=1000
echo ""
echo "[7/8] Llama3-8B-Instruct with mom2_n_samples=1000"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=mom2_samples_ablation/Llama3-8B-Instruct_mom2_1000.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json "$RESULTS_DIR/Llama3-8B-Instruct_mom2_1000_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Llama3-8B-Instruct_mom2_1000_summary.txt"

# Llama3-8B-Instruct with mom2_n_samples=10000
echo ""
echo "[8/8] Llama3-8B-Instruct with mom2_n_samples=10000"
echo "=========================================="
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=mom2_samples_ablation/Llama3-8B-Instruct_mom2_10000.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1
cp /home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json "$RESULTS_DIR/Llama3-8B-Instruct_mom2_10000_result.json"
python3 -m experiments.summarize_uns --file_path=/home/pg2860/anyedit/AnyEdit/output/MEMIT_ARE_Llama3-8B-Instruct_unke_result.json 2>&1 | tee "$RESULTS_DIR/Llama3-8B-Instruct_mom2_10000_summary.txt"

echo ""
echo "=========================================="
echo "MOM2 Samples Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Files:"
ls -la "$RESULTS_DIR"
