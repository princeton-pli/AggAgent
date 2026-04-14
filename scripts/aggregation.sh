STRATEGY=aggagent
MAX_WORKERS=5
K=4
MODEL=GLM-4.7-Flash
API_BASE=http://localhost:6000/v1
TASK=browsecomp
DIR=output/rollout/GLM-4.7-Flash/browsecomp

uv run python aggregation/aggregate.py --strategy $STRATEGY --max_workers $MAX_WORKERS --k $K --model $MODEL --api_base $API_BASE --task $TASK $DIR