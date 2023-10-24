source config

for dataset in ${DATASETS[@]}; do
  for part_method in ${PARTITIONING_METHODS[@]}; do
    for k in ${ks[@]}; do
      echo $dataset $part_method $k
      echo ./$BUILD_FOLDER/Partition ${DATA_PATH}/${dataset}_base1B.fbin $DATA_PATH/$dataset.partition $k $part_method
    done
  done
done
