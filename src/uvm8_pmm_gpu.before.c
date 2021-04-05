//uvm-pinning-setting
#define uvm-pinning_LOG
// #define uvm-pinning_ALGO
    // #define ALGO_SHUFFLE
    // #define ALGO_CUT
//uvm-pinning-func
size_t ratio_125(size_t count)
{
    return count >> 3;
}

size_t ratio_25(size_t count)
{
    return count >> 2;
}

size_t ratio_50(size_t count)
{
    return count >> 1;
}

size_t ratio_75(size_t count) 
{
    return (count >> 1) + (count >> 2);
}

size_t ratio_875(size_t count)
{
    return (count >> 1) + (count >> 2) + (count >> 3);
}

size_t ratio_9375(size_t count)
{
    return (count >> 1) + (count >> 2) + (count >> 3) + (count >> 4);
}

void thrashing_detection_raise(uvm_pmm_gpu_t * pmm) 
{
    pmm->thrashing_detection_raised = true;
    
    //uvm-pinning-algo
    uvm_spin_lock(&pmm->list_lock);

    struct list_head * head = &pmm->root_chunks.va_block_used;
    struct list_head * entry;

    size_t i, N;
    uvm_gpu_chunk_t * chunk;

    //uvm-pinning-tune
    N = ratio_50(pmm->available_chunks_count);

    i = 0;
    list_for_each(entry, head) {
        i++;
        if(i==N) break;
    }

    #ifdef ALGO_SHUFFLE
        // shuffle list
        head->prev->next = head->next;
        head->next->prev = head->prev;

        head->next = entry->next;
        entry->next->prev = head;

        entry->next = head;
        head->prev = entry;
    #elif(defined ALGO_CUT)
        // cut list
        pmm->reserve_entry = head->next;

        head->next = entry->next;
        entry->next->prev = head;

        entry->next = pmm->reserve_entry;
        pmm->reserve_entry->prev = entry;
    #endif

    uvm_spin_unlock(&pmm->list_lock);

    //uvm-pinning-log
    #ifdef uvm-pinning_LOG
        printk("uvm-pinning:%lu,cyclic_thrashing_detected,%u\n", NV_GETTIME(), pmm->available_chunks_count);
    #endif
}

void thrashing_detection_release(uvm_pmm_gpu_t * pmm)
{
    pmm->thrashing_detection_raised = false;
}
//