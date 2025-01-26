conda activate vllm
export HF_HOME=/fs/project/PAS1576/yiheng/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

datasets=(
 "nq_rear"
 "popqa"
 "musique"
 "2wikimultihopqa"
 "hotpotqa"
 "lveval"
 "narrativeqa_dev_10_doc"
)

for dataset in "${datasets[@]}"; do
  python src/apps/graph_statistics.py --dataset $dataset --llm vllm --llm_model meta-llama/Llama-3.3-70B-Instruct
  python src/apps/graph_statistics.py --dataset $dataset --llm openai --llm_model gpt-4o-mini
done