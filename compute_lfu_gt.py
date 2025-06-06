import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import heapq

max_cache_points = int(pow(10, 6))
total_points = 50000000

data_path = "/mnt/sdb/dynamic_index/linux_datasets/float32_gt100_SFR-Embedding-Code-400M_chunked_linux.bin"

with open(data_path, "rb") as fp:
    gt100_data = fp.read()

num_query = int.from_bytes(gt100_data[:4], byteorder="little")

k = int.from_bytes(gt100_data[4:8], byteorder="little")

gt100_arr = np.frombuffer(gt100_data[8:8 + (num_query * k * 4)], dtype=np.int32).reshape((-1, 100))

heap = []
num_intervals = 10
queries_per_interval = int(num_query / (num_intervals - 1))

for interval in range(num_intervals):
    start, end = interval * queries_per_interval, min((interval + 1) * queries_per_interval, total_points)
    slice = gt100_arr[start: end]
    freq_counter = Counter(slice.flatten())
    top_points = heapq.nlargest(max_cache_points, freq_counter.items(), key=lambda x: x[1])

    output_path = '/mnt/sdb/dynamic_index/linux_datasets/ground_truth_only/lfu_gt_' + str(interval)
    with open(output_path, "wb") as output:
        output.write(max_cache_points.to_bytes(4, 'big'))
        output.write(b''.join([int(point).to_bytes(4, 'big') for point, _ in top_points]))



