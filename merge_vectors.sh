cd /mnt/sdb/SPTAG/datasets/SPACEV1B/vectors.bin/

for i in {1..33}; do
	cat vectors_$i.bin >> vectors_merged.bin
done
