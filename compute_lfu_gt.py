import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import heapq
import matplotlib.pyplot as plt
from collections import defaultdict

max_cache_points = 8089 # num_points * 0.01 freq rate

vector_data_path = "/mnt/sdb/dynamic_index/linux_datasets/float32_dataset_SFR-Embedding-Code-400M_chunked_linux.bin"
data_path = "/mnt/sdb/dynamic_index/linux_datasets/float32_gt100_SFR-Embedding-Code-400M_chunked_linux.bin"
dim = 1024

with open(data_path, "rb") as fp:
    gt100_data = fp.read()

num_queries = int.from_bytes(gt100_data[:4], byteorder="little")

k = int.from_bytes(gt100_data[4:8], byteorder="little")

gt100_arr = np.frombuffer(gt100_data[8:8 + (num_queries * k * 4)], dtype=np.int32).reshape((-1, 100))

heap = []
num_intervals = 10
queries_per_interval = int(num_queries / (num_intervals - 1))
old_points = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray']
color_cache = {}
frequency_cache = defaultdict(int)

for interval in range(num_intervals - 1):
    curr_color = colors[interval]
    start, end = interval * queries_per_interval, min((interval + 1) * queries_per_interval, num_queries)

    slice = gt100_arr[start: end]
    freq_counter = Counter(slice.flatten())
    top_points = heapq.nlargest(max_cache_points, freq_counter.items(), key=lambda x: x[1])

    prefix = '/mnt/sdb/dynamic_index/linux_datasets/ground_truth_only/lfu_gt_' + str(interval) + '_' + str(max_cache_points)
    output_data_path = prefix + "_data.bin"
    output_id_path = prefix + "_ids.bin"
    original_data = open(vector_data_path, "rb")
    output_data = open(output_data_path, "wb")
    output_id = open(output_id_path, "wb")

    output_data.write(np.uint32(max_cache_points).tobytes())
    output_data.write(np.uint32(1024).tobytes())

    output_id.write(np.uint32(max_cache_points).tobytes())
    output_id.write(np.uint32(1).tobytes())

    float_size = np.int64(np.dtype(np.float32).itemsize)
    for node_id, _ in top_points:
        output_id.write(np.uint32(node_id).tobytes())
        
        vector_size_bytes = float_size * dim
        offset = 8 + float_size * np.int64(dim) * np.int64(node_id)

        original_data.seek(offset)
        vec = np.frombuffer(original_data.read(vector_size_bytes), dtype=np.float32, count=dim)
        output_data.write(vec.tobytes())
    
    point_ids = []
    point_freqs = []
    point_colors = []
    for point, _ in top_points:
        frequency_cache[point] += 1
        if point not in color_cache:
            color_cache[point] = curr_color
        
        point_ids.append(point)
        point_freqs.append(freq_counter[point])
        point_colors.append(color_cache[point])
    
    plot_output_path = '/mnt/sdb/dynamic_index/linux_datasets/ground_truth_only/lfu_gt_plot_' + str(interval)
    plt.figure(figsize=(20, 20))
    plt.scatter(point_ids, point_freqs, c=point_colors, s=10)
    plt.xlabel("point id")
    plt.ylabel("access frequency")

    plt.savefig(plot_output_path)

print(len(color_cache))
labels = [(i + 1) for i in range(num_intervals)]
points_of_each_frequency = [0 for _ in range(num_intervals)]
for interval_count in range(0, num_intervals):
    points_of_each_frequency[interval_count] = sum(1 for point, value in frequency_cache.items() if value == (interval_count + 1))

print(points_of_each_frequency)
plt.figure(figsize=(20, 20))
plt.bar(labels, points_of_each_frequency)
plt.savefig('/mnt/sdb/dynamic_index/linux_datasets/ground_truth_only/points_of_each_frequency')








