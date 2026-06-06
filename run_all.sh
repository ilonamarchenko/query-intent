#!/usr/bin/env bash
set -e
SWEEP=${1:-main}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${PYTHON:-python}

MODELS="bert-base-multilingual-cased xlm-roberta-base microsoft/mdeberta-v3-base xlm-roberta-large"

for DATASET in uk en both; do
    echo ">>> Dataset: $DATASET"
    for MODEL in $MODELS; do
        echo "  Model: $MODEL"
        $PYTHON src/models/run_experiments.py --dataset "$DATASET" --sweep "$SWEEP" --model "$MODEL"
    done
done

echo ">>> ukr-roberta-base on UK only"
$PYTHON src/models/train.py \
    --model  youscan/ukr-roberta-base \
    --output results/uk__ukr-roberta-base__ml256__lr2e-05__tsfull \
    --dataset uk

$PYTHON src/models/evaluate.py \
    --checkpoint results/uk__ukr-roberta-base__ml256__lr2e-05__tsfull/best \
    --output     results/uk__ukr-roberta-base__ml256__lr2e-05__tsfull/test_metrics.json \
    --dataset uk

echo "Done. Results -> results/summary.csv"
