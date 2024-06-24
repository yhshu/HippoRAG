DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Error: Dataset identifier not provided."
    exit 1
fi
rm "output/$DATASET_"*
rm "output/openie_$DATASET_"*
rm "exp/$DATASET_"*