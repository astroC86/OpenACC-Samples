module prefix_sum
    ! implements Blelloch Algorithm for prefix sum with swizzling
    use iso_fortran_env, only: int32

    integer(int32), parameter :: LOCAL_BUFFER_SIZE = 1024
    integer(int32), parameter :: MEMORY_BANKS      = 32
    integer(int32), parameter :: LOG_MEMORY_BANKS  = 5   ! log2(32)

    integer(int32), parameter :: MAX_BLOCK_DIM = 1024

    contains

    ! Swizzling function to avoid bank conflicts without extra storage
    ! that maps index to bank-conflict free location using XOR-based swizzle
    pure function swizzle_index(idx) result(swizzled)
        integer(int32), intent(in) :: idx
        integer(int32) :: swizzled
        integer(int32) :: bank_id, offset_within_bank
        
        ! Convert to 0-based for easier bit manipulation
        ! idx-1 gives 0-based index
        ! XOR with (idx-1) >> LOG_MEMORY_BANKS to swizzle
        bank_id = iand((idx - 1), (MEMORY_BANKS - 1))  ! Lower bits = bank
        offset_within_bank = ishft((idx - 1), -LOG_MEMORY_BANKS)  ! Upper bits = offset
        ! XOR the bank with a portion of the offset to distribute accesses
        swizzled = ior(ishft(ieor(bank_id, iand(offset_within_bank, (MEMORY_BANKS - 1))), LOG_MEMORY_BANKS), offset_within_bank)        
        swizzled = swizzled + 1
    end function swizzle_index

    subroutine get_time_us(time)
        real(kind=8), intent(out) :: time
        integer(kind=8) :: count, count_rate
        
        call system_clock(count, count_rate)
        time = real(count, 8) * 1000000.0d0 / real(count_rate, 8)
    end subroutine

    subroutine add_block_sum(vector, block_sum, N, NUM_BLOCKS, NUM_THREADS)  
        use cudafor      
        implicit none
        integer(int32), device, intent(inout) :: vector(:)
        integer(int32), device, intent(in) :: block_sum(:)
        integer(int32), intent(in) :: NUM_BLOCKS
        integer(int32), intent(in) :: NUM_THREADS
        integer(int32), intent(in) :: N
        
        ! Local variables
        integer(int32) :: block_id, i
        integer(int32) :: idx

        !$acc parallel loop gang vector_length(NUM_THREADS) copy(vector)
        do block_id = 1, NUM_BLOCKS            
            !$acc loop vector private(idx)
            do i = 1, NUM_THREADS
                idx  = 2 * (block_id - 1) * NUM_THREADS + i
                if (idx <= N) then
                    vector(idx) = vector(idx) + block_sum(block_id)
                    if (idx + NUM_THREADS <= N) then
                        vector(idx + NUM_THREADS) = vector(idx + NUM_THREADS ) + block_sum(block_id)
                    end if
                end if
            end do
        end do
        
    end subroutine add_block_sum

    subroutine scan(d_in ,d_out, block_sums, size, NUM_BLOCKS, NUM_THREADS, SMEM_SIZE, MAX_ELE_PER_BLOCK)
        use cudafor
        implicit none
        
        integer(int32), intent(in)    :: size
        integer(int32), intent(in)    :: SMEM_SIZE, NUM_THREADS, MAX_ELE_PER_BLOCK
        integer(int32), intent(in)    :: NUM_BLOCKS

        integer(int32), device, intent(in   ) :: d_in(:)
        integer(int32), device, intent(inout) :: d_out(:)
        integer(int32), device, intent(inout) :: block_sums(:)
        
        integer(int32) :: local_buffer(1024)
        integer(int32) :: tid, ai, bi, idx, offset, mask
        integer(int32) :: block_id
        integer(int32) :: tmp
        integer(int32) :: swizzled_ai, swizzled_bi

        !$acc parallel loop gang num_gangs(NUM_BLOCKS) vector_length(NUM_THREADS) &
        !$acc private(local_buffer,offset,mask,swizzled_ai,swizzled_bi) &
        !$acc firstprivate(max_ele_per_block, size,SMEM_SIZE)   &
        !$acc copyout(block_sums) present(d_in ,d_out)
        do block_id = 1, NUM_BLOCKS
            
            !$acc cache(local_buffer(:))
            
            local_buffer(:) = 0

            ! Load data with swizzled indices
            !$acc loop vector private(idx,ai,bi,swizzled_ai,swizzled_bi)
            do tid = 1, NUM_THREADS
                ai = tid 
                bi = tid + NUM_THREADS
                
                idx = (block_id - 1) * MAX_ELE_PER_BLOCK + tid

                if (idx <= size) then
                    swizzled_ai = swizzle_index(ai)
                    local_buffer(swizzled_ai) = d_in(idx)
                    if (idx + NUM_THREADS <= size) then
                        swizzled_bi = swizzle_index(bi)
                        local_buffer(swizzled_bi) = d_in(idx + NUM_THREADS)
                    end if
                end if
            end do

            ! Up-sweep (reduce) phase
            offset = 1
            mask   = MAX_ELE_PER_BLOCK / 2
            do while (mask > 0)
                !$acc loop vector private(ai,bi,swizzled_ai,swizzled_bi)
                do tid = 0, mask - 1
                    ai = offset * ((tid * 2) + 1) - 1
                    bi = offset * ((tid * 2) + 2) - 1
                    ! Convert to 1-based and swizzle
                    ai = ai + 1
                    bi = bi + 1
                    swizzled_ai = swizzle_index(ai)
                    swizzled_bi = swizzle_index(bi)
                    local_buffer(swizzled_bi) = local_buffer(swizzled_bi) + local_buffer(swizzled_ai)
                end do
                offset = offset * 2
                mask = mask / 2
            end do
            
            ! Store block sum and reset last element
            swizzled_bi = swizzle_index(MAX_ELE_PER_BLOCK)
            block_sums(block_id) = local_buffer(swizzled_bi)
            local_buffer(swizzled_bi) = 0
            
            ! Down-sweep phase
            mask = 1
            offset = offset / 2

            do while (mask < MAX_ELE_PER_BLOCK)
            
                !$acc loop vector private(ai, bi, tmp, swizzled_ai, swizzled_bi)
                do tid = 0, mask - 1
                    ai = offset * ((tid * 2) + 1) - 1
                    bi = offset * ((tid * 2) + 2) - 1
                    ! Convert to 1-based and swizzle
                    ai = ai + 1
                    bi = bi + 1
                    swizzled_ai = swizzle_index(ai)
                    swizzled_bi = swizzle_index(bi)
                    
                    tmp = local_buffer(swizzled_ai)
                    local_buffer(swizzled_ai) = local_buffer(swizzled_bi)
                    local_buffer(swizzled_bi) = local_buffer(swizzled_bi) + tmp
                end do

                mask = mask * 2
                offset = offset / 2
            end do
        
            ! Store results with swizzled indices
            !$acc loop vector private(ai, bi, idx, swizzled_ai, swizzled_bi)
            do tid = 1, NUM_THREADS
                idx = (block_id - 1) * MAX_ELE_PER_BLOCK + tid
                if (idx <= size) then
                    ai = tid
                    bi = tid + NUM_THREADS
                    swizzled_ai = swizzle_index(ai)
                    swizzled_bi = swizzle_index(bi)
                    d_out(idx) = local_buffer(swizzled_ai)
                    if (idx + NUM_THREADS <= size) then
                        d_out(idx + NUM_THREADS) = local_buffer(swizzled_bi)
                    end if
                end if
            end do

        end do
        
    end subroutine scan

    recursive subroutine prefix_sum_openacc_v2(d_in, d_out, num_elems)
        use cudafor
        implicit none

        integer(int32), device, intent(in) :: d_in(:)
        integer(int32), device, intent(out) :: d_out(:)
        integer(int32), intent(in)  :: num_elems

        integer(int32) :: block_sz, max_elems_per_block, grid_sz, shmem_sz
        
        integer(int32), device, allocatable :: d_block_sums(:)
        integer(int32), device, allocatable :: d_dummy_blocks_sums(:)
        integer(int32), device, allocatable :: d_in_block_sums(:)

        integer(int32) :: ierr, i
        
        block_sz = MAX_BLOCK_DIM / 2
        max_elems_per_block = 2 * block_sz

        grid_sz = (num_elems + max_elems_per_block - 1)/ max_elems_per_block
        
        ! With swizzling, shared memory size equals max_elems_per_block (no padding needed)
        shmem_sz = max_elems_per_block

        allocate(d_block_sums(grid_sz), stat=ierr)
        !$acc kernels
        d_block_sums(:) = 0
        !$acc end kernels 
        
        call scan( d_in, d_out, d_block_sums, num_elems, grid_sz, block_sz, shmem_sz, max_elems_per_block)

        if (grid_sz <= max_elems_per_block) then
            allocate(d_dummy_blocks_sums(1), stat=ierr)
            call scan(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz, int(1, int32), block_sz, shmem_sz, max_elems_per_block)
            deallocate(d_dummy_blocks_sums)
        else
            allocate(d_in_block_sums(grid_sz), stat=ierr)
            
            !$acc parallel loop present(d_in_block_sums, d_block_sums)
            DO i = 1, grid_sz
                d_in_block_sums(i) = d_block_sums(i)
            end dO
            !$acc end parallel loop

            call prefix_sum_openacc_v2(d_in_block_sums, d_block_sums, grid_sz)
            deallocate(d_in_block_sums)
        end if
                
        call add_block_sum(d_out, d_block_sums, num_elems, grid_sz, block_sz)
        deallocate(d_block_sums)
    end subroutine prefix_sum_openacc_v2

    subroutine prefix_sum_serial(vector, size)
        implicit none
        
        ! Arguments
        integer(int32), intent(inout) :: vector(:)
        integer(int32), intent(in) :: size
        
        ! Local variables
        integer(int32) :: acc, temp
        integer :: i
        
        acc = 0
        do i = 1, size  ! Note: Fortran uses 1-based indexing
            temp = vector(i)
            vector(i) = acc
            acc = acc + temp
        end do
        
    end subroutine prefix_sum_serial

end module prefix_sum

program test_prefix_sum
    use prefix_sum
    implicit none
    
    ! Test parameters
    integer(int32), parameter :: TEST_SIZE =  5002368*2*2*2*2*2
    integer(int32), allocatable :: vec_serial(:), vec_parallel(:), vec_parallel_out(:)
    integer(int32) :: i, max_diff
    logical :: is_correct
    real(kind=8) :: start_time, end_time
    real(kind=8) :: pstart_time, pend_time
    integer :: diff_count = 0
    
    allocate(vec_serial(TEST_SIZE))
    allocate(vec_parallel(TEST_SIZE))
    allocate(vec_parallel_out(TEST_SIZE))
    
    do i = 1, TEST_SIZE
        vec_serial(i) = i
        vec_parallel(i) = vec_serial(i)
    end do
    
    !$acc data copyin(vec_parallel) create(vec_parallel_out) copyout(vec_parallel_out)

    ! Warm up parallel version
    call prefix_sum_openacc_v2(vec_parallel,vec_parallel_out, TEST_SIZE)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_openacc_v2(vec_parallel,vec_parallel_out, TEST_SIZE)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_openacc_v2(vec_parallel,vec_parallel_out, TEST_SIZE)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_openacc_v2(vec_parallel,vec_parallel_out, TEST_SIZE)
    vec_parallel(:) = vec_serial(:)

    call get_time_us(pstart_time)
    !$acc host_data use_device(vec_parallel,vec_parallel_out)
    call prefix_sum_openacc_v2(vec_parallel,vec_parallel_out, TEST_SIZE)
    !$acc end host_data 
    call get_time_us(pend_time)
    !$acc end data

    ! Run serial version
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_serial(vec_serial, TEST_SIZE)
    vec_serial(:) = vec_parallel(:)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_serial(vec_serial, TEST_SIZE)
    vec_serial(:) = vec_parallel(:)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_serial(vec_serial, TEST_SIZE)
    vec_serial(:) = vec_parallel(:)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_serial(vec_serial, TEST_SIZE)
    vec_serial(:) = vec_parallel(:)
    vec_parallel(:) = vec_serial(:)
    call prefix_sum_serial(vec_serial, TEST_SIZE)
    vec_serial(:) = vec_parallel(:)

    call get_time_us(start_time)
    call prefix_sum_serial(vec_serial, TEST_SIZE)
    call get_time_us(end_time)
    print '(A,F10.2,A)', 'Serial version time  : ', end_time - start_time, ' microseconds'
    print '(A,F10.2,A)', 'Parallel version time: ', pend_time - pstart_time, ' microseconds'
    
    ! Check correctness
    is_correct = .true.
    max_diff = 0
    
    do i = 1, TEST_SIZE
        if (vec_serial(i) /= vec_parallel_out(i)) then
            is_correct = .false.
            max_diff = max(max_diff, abs(vec_serial(i) - vec_parallel_out(i)))
        end if
    end do
    
    ! Print results
    print *, '=== Test Results ==='
    print *, 'Test size:', TEST_SIZE
    if (is_correct) then
        print *, 'Status: PASSED - Results match exactly'
    else
        print *, 'Status: FAILED - Results differ'
        print *, 'Maximum difference:', max_diff
        
        ! Print first few differences found
        print *, 'First few differences (idx: serial, parallel):'
        do i = 1, TEST_SIZE
            if (vec_serial(i) /= vec_parallel_out(i)) then
                if (diff_count == 0) print '(A,I6,A,I12,A,I12)', 'Before ', i - 1, ': ', vec_serial(i -1), ', ', vec_parallel_out(i -1)
                print '(A,I6,A,I12,A,I12)', 'Index ', i, ': ', vec_serial(i), ', ', vec_parallel_out(i)
                diff_count = diff_count + 1
                if (diff_count >= 5) exit
            end if
        end do
    end if
    
    ! Clean up
    deallocate(vec_serial)
    deallocate(vec_parallel)
    deallocate(vec_parallel_out)
    
end program test_prefix_sum
