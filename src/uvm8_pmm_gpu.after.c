    //uvm-pinning-algo
#if(defined uvm-pinning_LOG || defined uvm-pinning_ALGO)
        if (chunk && chunk->va_block) {
            size_t block_evict_cnt = ++chunk->va_block->eviction_count;
        #ifdef uvm-pinning_ALGO

            // Runs only on the very first page eviction.
            if (!pmm->available_chunks_count) { 
                struct list_head * it;
                size_t N = 0;

                list_for_each(it, &pmm->root_chunks.va_block_used) { N++; }
                pmm->available_chunks_count = N;
            }

            size_t detection_evict_cnt = pmm->eviction_count_curr;
    #ifdef ALGO_CUT
    if (!pmm->thrashing_detection_raised) {
    #endif
            if (block_evict_cnt == detection_evict_cnt) {
                pmm->eviction_peers_curr++;
                //uvm-pinning-tune
                if (!pmm->thrashing_detection_raised &&
                    pmm->eviction_peers_prev &&
                    pmm->eviction_peers_curr > ratio_75(pmm->eviction_peers_prev)) {
                    thrashing_detection_raise(pmm);
                }
            }
            else if (block_evict_cnt - 1 == detection_evict_cnt) {
                pmm->eviction_peers_next++;
                if (pmm->eviction_peers_next > ratio_125(pmm->eviction_peers_curr)) {
                    pmm->eviction_count_curr++;
                    pmm->eviction_peers_prev = pmm->eviction_peers_curr;
                    pmm->eviction_peers_curr = pmm->eviction_peers_next;
                    pmm->eviction_peers_next = 0;
                    thrashing_detection_release(pmm);
                }
            }
            else if (block_evict_cnt + 1 == detection_evict_cnt) {
                pmm->eviction_peers_prev++;
            }
        #endif
    #ifdef ALGO_CUT
        }
    #endif
        //uvm-pinning-log
        #ifdef uvm-pinning_LOG
            printk("uvm-pinning:%lu,%08x,%08x,%u\n", NV_GETTIME(), chunk->va_block->start, chunk->address,  chunk->va_block->eviction_count);
        #endif
        }
#endif