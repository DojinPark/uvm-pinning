#define CUDA_MAX_BLOCK_DIM 65535
#define CUDA_MAX_THREADS_PER_BLOCK 1024
int check_numblocks(size_t numblock);
void occupy_gpu_space(float free_space_in_mb);
void occupy_gpu_space(char * free_space_in_mb_str);
void set_envs(size_t * data_len, int dtype_size, char * arg_data_size_in_mb, char * arg_free_space_in_mb, bool * is_prefetch, char * arg_is_prefetch);
void gpu_timer_set();
void gpu_timer_pause();
void gpu_timer_record(int argc, char ** argv);
