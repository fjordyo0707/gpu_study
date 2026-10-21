# Roofline Report

Memory bandwidth ceiling: 484.0 GB/s

FP32 compute ceiling: 10600.0 GFLOP/s

| Kernel | AI (FLOP/byte) | Measured GFLOP/s | Memory roof | Attainable roof | Classification | Notes |
| ------ | --------------: | ---------------: | ----------: | --------------: | -------------- | ----- |
| vector_add | 0.083 | 31.970 | 40.333 | 40.333 | near memory roof | week01/01_vector_add/README.md proper microbenchmark baseline; N=16777216; threads=256; avg=0.524786 ms; GFLOP/s=N/time; AI=1/12 |
| matmul_naive | 170.667 | 448.103 | 82602.667 | 10600.000 | below compute roof; investigate overhead | week01/02_matrix_multiply/std_record.log; N=1024; ideal compulsory-traffic AI=N/6 assumes each matrix transferred once; actual DRAM AI unknown |
| matmul_tiled | 170.667 | 1006.580 | 82602.667 | 10600.000 | below compute roof; investigate overhead | week01/02_matrix_multiply/std_record.log; N=1024; tile=16; ideal compulsory-traffic AI=N/6 assumes each matrix transferred once; actual DRAM AI unknown |
| matmul_register_blocking | 170.667 | 1373.441 | 82602.667 | 10600.000 | below compute roof; investigate overhead | week01/02_matrix_multiply/std_record.log; N=1024; tile=16; outputs/thread=2; ideal compulsory-traffic AI=N/6 assumes each matrix transferred once; actual DRAM AI unknown |
| matmul_cublas | 170.667 | 7559.304 | 82602.667 | 10600.000 | near compute roof | week01/02_matrix_multiply/std_record.log; N=1024; cublasSgemm; ideal compulsory-traffic AI=N/6; actual DRAM AI unknown |
| arithmetic_intensity_fma_1 | 0.250 | 89.853 | 121.000 | 121.000 | near memory roof | week03/01_arithmetic_intensity/std_record.log; N=16777216; AI=2*1/8; PASS |
| arithmetic_intensity_fma_4 | 1.000 | 359.160 | 484.000 | 484.000 | near memory roof | week03/01_arithmetic_intensity/std_record.log; N=16777216; AI=2*4/8; PASS |
| arithmetic_intensity_fma_16 | 4.000 | 1436.489 | 1936.000 | 1936.000 | near memory roof | week03/01_arithmetic_intensity/std_record.log; N=16777216; AI=2*16/8; PASS |
| arithmetic_intensity_fma_64 | 16.000 | 4568.001 | 7744.000 | 7744.000 | near memory roof | week03/01_arithmetic_intensity/std_record.log; N=16777216; AI=2*64/8; PASS |
| arithmetic_intensity_fma_256 | 64.000 | 6698.538 | 30976.000 | 10600.000 | near compute roof | week03/01_arithmetic_intensity/std_record.log; N=16777216; AI=2*256/8; PASS |
| arithmetic_intensity_fma_1024 | 256.000 | 7652.792 | 123904.000 | 10600.000 | near compute roof | week03/01_arithmetic_intensity/std_record.log; N=16777216; AI=2*1024/8; PASS |
| memory_stream |  |  |  |  | TODO: fill missing values | week03/03_memory_vs_compute; planned AI=4/16; no recorded run available |
| compute_chain_512 |  |  |  |  | TODO: fill missing values | week03/03_memory_vs_compute; planned AI=2*512/8; no recorded run available |
