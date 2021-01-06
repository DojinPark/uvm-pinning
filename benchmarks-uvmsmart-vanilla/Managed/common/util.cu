#include <iostream>
#include <fstream>
#include <string>
#include "util.h"

#define BYTES_IN_GB 1073741824
#define BYTES_IN_MB 1048576

bool parse_bool(char * str) {
    return std::stoi(str)? 1 : 0;
}

size_t parse_data_len(char * arg_data_size_in_mb, int dtype_size) {
    size_t bytes = BYTES_IN_MB * std::stof(arg_data_size_in_mb);
    size_t len = bytes / dtype_size;
    std::cout << bytes / BYTES_IN_MB << " MB with " << dtype_size << " Bytes per data instance. The number of data instances is " << len << std::endl;
    return len;
}

int check_numblocks(size_t numblock) {
    if (numblock > 65535) return 65535;
    return (int)numblock;
}

void occupy_gpu_space(float free_space_in_mb) {
    if (free_space_in_mb == 0) return;

    // Get currently free memory space. 
    size_t curr_bytes, dummy;
    cudaMemGetInfo(&curr_bytes, &dummy);
    size_t occupy_bytes = curr_bytes - (size_t)(free_space_in_mb * BYTES_IN_MB);

    // Occupy memory space, leaving only specified space for memory size simulation
    char * occupy_ptr;
    cudaMalloc(&occupy_ptr, occupy_bytes);

    // Print memory size simulation info.
    size_t free_bytes;
    cudaMemGetInfo(&free_bytes, 0);
    // printf("%u %u %u\n", curr_bytes, occupy_bytes, free_bytes);
    printf("GPU Memory Manipulation: Occupied=%.1fMB, Free Space=%.1fMB\n", ((float)occupy_bytes / BYTES_IN_MB), ((float)free_bytes / BYTES_IN_MB));
}

void occupy_gpu_space(char * free_space_in_mb_str) {
    occupy_gpu_space(std::stof(free_space_in_mb_str));
}

void set_envs(size_t * data_len, int dtype_size, char * arg_data_size_in_mb, char * arg_free_space_in_mb, bool * is_prefetch, char * arg_is_prefetch) {
    // std::cout << arg_data_size_in_mb << " " << arg_free_space_in_mb << " " << arg_is_prefetch << std::endl;

    printf("⸢------------------ Setting Test Environment ----------------------⸣\n");
    if (arg_data_size_in_mb)
        *data_len = parse_data_len(arg_data_size_in_mb, dtype_size);
    if (arg_free_space_in_mb)
        occupy_gpu_space(arg_free_space_in_mb);
    if (arg_is_prefetch)
        *is_prefetch = parse_bool(arg_is_prefetch);
    printf("⸤ ------------------------------------------------------------------⸥\n");
}

cudaEvent_t start, stop;
float gpu_time = 0;
void gpu_timer_set() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

void gpu_timer_pause() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    gpu_time += ms;
}

void gpu_timer_record(int argc, char ** argv) {
    std::ofstream record;
    record.open("gpu_time.txt", std::ios::app);
    for (int i=0; i<argc; i++)
        record << argv[i] << '\t';
    record << gpu_time << std::endl;
    printf("⸢----------------------⸣\n");
    printf("gpu time: %f ms\n", gpu_time);
    printf("⸤----------------------⸥\n");
    record.close();
}
