#!/bin/bash

data_type='int8'
data='/mnt/sdb/vectors_merged.bin'
query='/mnt/sdb/query.bin'
index_prefix='/mnt/sdb/local_index'
result=/mnt/sdb/results/results_
inserts=20000
active_window=10000
cons_int=500
thr=64
index=${index_prefix}.after-streaming-act${active_window}-cons${cons_int}-max${inserts}
gt_file=/mnt/sdb/gt100_learn-act${active}-cons${cons_int}-max${inserts}
log=/mnt/sdb/dynamic_index/streaming_results/results-act${active}-cons${cons_int}-max${inserts}

/mnt/sdb/DiskANN/build/apps/test_streaming_scenario --data_type ${data_type} --dist_fn l2 --data_path ${data} --index_path_prefix ${index_prefix} -R 64 -L 300 --alpha 1.2 --insert_threads ${thr} --consolidate_threads ${thr}  --max_points_to_insert ${inserts} --active_window ${active_window} --consolidate_interval ${cons_int} --start_point_norm 0.2;

/mnt/sdb/DiskANN/build/apps/utils/compute_groundtruth --data_type ${data_type} --dist_fn l2 --base_file ${data} --query_file ${query} --K 100 --gt_file ${gt_file} --start_offset 9501 --end_offset 20000

/mnt/sdb/DiskANN/build/apps/search_memory_index --data_type ${data_type} --dist_fn l2 --index_path_prefix ${index} --result_path ${result} --query_file ${query} --gt_file ${gt_file} -K 10 -L 20 40 60 80 100 -T ${thr} --dynamic true --tags 1 > ${log}