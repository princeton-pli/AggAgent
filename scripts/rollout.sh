MODEL=MiniMaxAI/MiniMax-M2.5
DATASETS=(browsecomp browsecomp-plus hle deepsearchqa healthbench researchrubrics)
ROLL_OUT_COUNT=8
MAX_WORKERS=3
API_BASE=http://localhost:6000/v1

export SEARCH_SERVER_URL="http://localhost:8765"
uv run python rollout/tools/serve_search.py --host 0.0.0.0 --port 8765 --workers 3 &

for DATASET in "${DATASETS[@]}"; do
    OUTPUT_DIR=output/${MODEL##*/}/$DATASET
    echo "=== Running dataset: $DATASET ==="
    if [ "$DATASET" = "browsecomp-plus" ]; then
        export SEARCHER_TYPE=faiss
        export INDEX_PATH="data/browsecomp-plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl"
        export SEARCH_MODEL_NAME=Qwen/Qwen3-Embedding-8B
        export SEARCH_NORMALIZE=true
        export SEARCH_DATASET_NAME="data/browsecomp-plus/corpus"
        export SEARCH_K=5
        export SNIPPET_MAX_TOKENS=512
        export SNIPPET_TOKENIZER_PATH=Qwen/Qwen3-0.6B
    fi
    uv run python rollout/run_multi_react.py --model $MODEL --output_dir $OUTPUT_DIR --dataset $DATASET --roll_out_count $ROLL_OUT_COUNT --max_workers $MAX_WORKERS --api_base $API_BASE
done