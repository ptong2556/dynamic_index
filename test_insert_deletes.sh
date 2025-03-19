data_type='int8'
data='/mnt/sdb/vectors_merged.bin'
query='/mnt/sdb/query.bin'
index_prefix='/mnt/sdb/local_index'
result=/mnt/sdb/results/results_
deletes=50000000
inserts=100000000
deletes_after=50000000
pts_per_checkpoint=1000000
begin=0
thr=64
index=${index_prefix}.after-delete-del${deletes}-${inserts}
gt_file=/mnt/sdb/gt100_learn-conc-${deletes}-${inserts}
log=/mnt/sdb/dynamic_index/log_results/results_${deletes}-${inserts}

/mnt/sdb/DiskANN/build/apps/test_insert_deletes_consolidate --data_type ${data_type} --dist_fn l2 --data_path ${data} --index_path_prefix ${index_prefix} -R 64 -L 300 --alpha 1.2 -T ${thr} --points_to_skip 0 --max_points_to_insert ${inserts} --beginning_index_size ${begin} --points_per_checkpoint ${pts_per_checkpoint} --checkpoints_per_snapshot 0 --points_to_delete_from_beginning ${deletes} --start_deletes_after ${deletes_after} --start_point_norm 0.2

# --base_file {data}
# --start_offset 50000001 --end_offset 100000000
/mnt/sdb/DiskANN/build/apps/utils/compute_groundtruth --data_type ${data_type} --dist_fn l2 --base_file ${data} --query_file ${query} --K 100 --gt_file ${gt_file} --start_offset 50000001 --end_offset 100000000

/mnt/sdb/DiskANN/build/apps/search_memory_index  --data_type ${data_type} --dist_fn l2 --index_path_prefix ${index} --result_path ${result} --query_file ${query}  --gt_file ${gt_file}  -K 10 -L 20 40 60 80 100 -T ${thr} --dynamic true --tags 1 > ${log}