#include <iostream>
#include "util.h"

int main() {
    int dtype_size = 4;
    char * arg_data_size_in_mb = "100";
    char * arg_free_space_in_mb = "200";

    size_t data_len;
    set_envs(&data_len, dtype_size,arg_data_size_in_mb, arg_free_space_in_mb, 0, 0);
    
}