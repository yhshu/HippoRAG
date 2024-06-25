DATASET=$1
RETRIEVER=facebook_contriever_mean
if [ -z "$DATASET" ]; then
    echo "Error: Dataset identifier not provided."
    exit 1
fi
rm "output/$DATASET_"*
rm "output/openie_$DATASET_"*
rm "exp/$DATASET_"*
rm "data/lm_vectors/${RETRIEVER}/vecs_"*
rm "data/lm_vectors/${RETRIEVER}/nearest_neighbor_"*
rm "data/lm_vectors/${RETRIEVER}/encoded_strings.txt"