    //uvm-pinning-struct
    size_t eviction_count_curr;
    size_t eviction_peers_prev;
    size_t eviction_peers_curr;
    size_t eviction_peers_next;
    bool thrashing_detection_raised;
    size_t available_chunks_count;
    struct list_head * reserve_entry;
    //