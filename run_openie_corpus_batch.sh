#!/bin/bash

# Check if two arguments (start and end numbers) are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <start_number> <end_number>"
  exit 1
fi

# Get the start and end numbers from arguments
start_num=$1
end_num=$2

# Loop through the range from start_num to end_num
for num in $(seq $start_num $end_num); do
  # Run the Python script with the generated file path
  python src/openie_corpus_batch.py --corpus "data/corpus_batch/musique_${num}_corpus.json"
done
