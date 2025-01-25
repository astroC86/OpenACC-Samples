module prefix_sum
    ! implements Blelloch Algorithm for prefix sum
    use iso_fortran_env, only: int32, int32

    integer(int32), parameter :: LOCAL_BUFFER_SIZE = 1024
    integer(int32), parameter :: MEMORY_BANKS      = 32



    integer(int32), parameter :: MAX_BLOCK_DIM = 1024


    contains


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

        !$acc parallel loop gang  vector_length(NUM_THREADS) copy(vector)
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
        
        integer(int32) :: local_buffer( (MAX_BLOCK_DIM +  (((MAX_BLOCK_DIM - 1) / MEMORY_BANKS))) )
        integer(int32) :: tid, ai, bi, idx, offset, mask
        integer(int32) :: block_id
        integer(int32) :: tmp

        
        !$acc parallel loop gang num_gangs(NUM_BLOCKS) vector_length(NUM_THREADS) &
        !$acc private(local_buffer,offset,mask) firstprivate(max_ele_per_block, size,SMEM_SIZE)   &
        !$acc copyout(block_sums) present(d_in ,d_out)
        do block_id = 1, NUM_BLOCKS
            
            !$acc cache(local_buffer(:))
            
            local_buffer(:) = 0

            !$acc loop vector private(idx,ai,bi)
            do tid = 1, NUM_THREADS
                ai = tid 
                bi = tid + NUM_THREADS
                
                idx = (block_id - 1) * MAX_ELE_PER_BLOCK + tid

                if (idx <= size) then
                  local_buffer(ai + ((ai - 1)/MEMORY_BANKS)) = d_in(idx)
                  if (idx + NUM_THREADS <= size) then
                    local_buffer(bi + ((bi - 1)/MEMORY_BANKS)) = d_in(idx + NUM_THREADS)
                  end if
                end if
            end do

            offset = 1
            mask   = MAX_ELE_PER_BLOCK / 2
            do while (mask > 0)
                !$acc loop vector private(ai,bi)
                do tid = 0, mask - 1
                    ai = offset * ((tid * 2) + 1) - 1
                    bi = offset * ((tid * 2) + 2) - 1
                    ai = ai + (ai / MEMORY_BANKS) + 1
                    bi = bi + (bi / MEMORY_BANKS) + 1
                    local_buffer(bi) = local_buffer(bi) + local_buffer(ai)
                end do
                offset = offset * 2
                mask = mask / 2
            end do
            
            block_sums(block_id) = local_buffer(MAX_ELE_PER_BLOCK + (MAX_ELE_PER_BLOCK - 1)/MEMORY_BANKS )
            local_buffer(MAX_ELE_PER_BLOCK + (MAX_ELE_PER_BLOCK - 1)/MEMORY_BANKS) = 0
            
            mask = 1
            offset = offset / 2

            do while (mask < MAX_ELE_PER_BLOCK)
            
                !$acc loop vector private(ai, bi, tmp)
                do tid = 0, mask - 1
                    ai = offset * ((tid * 2) + 1) - 1
                    bi = offset * ((tid * 2) + 2) - 1
                    ai = ai + (ai/MEMORY_BANKS) + 1
                    bi = bi + (bi/MEMORY_BANKS) + 1
                    
                    tmp = local_buffer(ai)
                    local_buffer(ai) = local_buffer(bi)
                    local_buffer(bi) = local_buffer(bi) + tmp
                end do

                mask = mask * 2
                offset = offset / 2
            end do
        
            !$acc loop vector private(ai, bi, idx)
            do tid = 1, NUM_THREADS
                idx = (block_id - 1) * MAX_ELE_PER_BLOCK + tid
                if (idx <= size) then
                    ai = tid
                    bi = tid + NUM_THREADS                    
                    d_out(idx) = local_buffer(ai + ((ai - 1)/MEMORY_BANKS))
                    if (idx + NUM_THREADS <= size) then
                        d_out(idx + NUM_THREADS) = local_buffer(bi + ((bi - 1)/MEMORY_BANKS))
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
        shmem_sz = max_elems_per_block +  (((max_elems_per_block - 1)/MEMORY_BANKS))

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

!Serial version time  : 1035909.20 microseconds
!Parallel version time:   49392.90 microseconds

program test_prefix_sum
    use prefix_sum
    implicit none
    
    ! Test parameters
    integer(int32), parameter :: TEST_SIZE = 5002368!*2*2*2*2*2*2*2*2
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
        vec_serial(i) = i !mod(i * 17 + 13, 100_int32)  ! Some arbitrary values
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
                if (diff_count == 0) print '(A,I6,A,I12,A,I12)', 'Befor ', i - 1, ': ', vec_serial(i -1), ', ', vec_parallel_out(i -1)
                print '(A,I6,A,I12,A,I12)', 'Index ', i, ': ', vec_serial(i), ', ', vec_parallel_out(i)
                diff_count = diff_count + 1
                if (diff_count >= 5) exit
            end if
        end do
    end if
    
    ! Print sample of results
    !print *, 'Sample of results (first 16 elements):'
    !print *, 'Index    Serial      Parallel'
    !do i = 1, min(40_int32, TEST_SIZE)
    !    print '(I5,2I12)', i, vec_serial(i), vec_parallel_out(i)
    !end do


    ! Clean up
    deallocate(vec_serial)
    deallocate(vec_parallel)
    deallocate(vec_parallel_out)
    
end program test_prefix_sum

    ! with shared memory cache conflict
    !subroutine prefix_sum_full(vector, block_sum, NUM_BLOCKS, size)
    !    implicit none
    !    ! Arguments
    !    integer(int32), intent(inout) :: vector(:)
    !    integer(int32), intent(out) :: block_sum(:)
    !    integer(int32), intent(in) :: NUM_BLOCKS
    !    integer(int32), intent(in) :: size
    !    
    !    ! Local variables
    !    integer(int32) :: block_id, i, offset
    !    integer(int32) :: begin_idx, temp
    !    integer(int32) :: local_buffer(LOCAL_BUFFER_SIZE)
    !    integer(int32) :: left_idx, right_idx, dest_idx,x,y
    !    
    !    !$acc parallel loop gang vector_length(LOCAL_BUFFER_SIZE/2) private(local_buffer) present(block_sum)
    !    do block_id = 1, NUM_BLOCKS
    !        begin_idx = (block_id - 1) * LOCAL_BUFFER_SIZE
    !        
    !        !$acc cache(local_buffer(1:LOCAL_BUFFER_SIZE))
    !        
    !        !$acc loop vector
    !        do i = 1, LOCAL_BUFFER_SIZE
    !            if (i + begin_idx <= size) then
    !                local_buffer(i) = vector(i + begin_idx)
    !            else
    !                local_buffer(i) = 0
    !            end if
    !        end do
    !        
    !        
    !        offset = 1
    !        do while (offset < LOCAL_BUFFER_SIZE)
    !            !$acc loop vector
    !            do i = offset, LOCAL_BUFFER_SIZE - 1, 2 * offset
    !                local_buffer(i + offset) = local_buffer(i + offset) + local_buffer(i)
    !            end do
    !            offset = offset * 2
    !        end do
    !        
    !        block_sum(block_id) = local_buffer(LOCAL_BUFFER_SIZE)
    !        local_buffer(LOCAL_BUFFER_SIZE) = 0
!
!
    !        offset = LOCAL_BUFFER_SIZE / 2
    !        do while (offset > 0)
    !            !$acc loop vector
    !            do i = offset, LOCAL_BUFFER_SIZE - 1, 2 * offset
    !                temp = local_buffer(i)
    !                local_buffer(i) = local_buffer(i + offset)
    !                local_buffer(i + offset) = local_buffer(i + offset) + temp
    !            end do
    !            offset = offset / 2
    !        end do
    !        
    !        !$acc loop vector
    !        do i = 1, LOCAL_BUFFER_SIZE
    !            if (i + begin_idx <= size) then
    !                vector(i + begin_idx) = local_buffer(i)
    !            end if
    !        end do
    !    end do
    !    
    !end subroutine prefix_sum_full

    !recursive subroutine prefix_sum_openacc(vector, size)
    !    implicit none
    !    integer(int32), intent(inout) :: vector(:)
    !    integer(int32), intent(in) :: size
    !    integer(int32) :: NUM_BLOCKS
    !    integer(int32), allocatable :: block_sum(:)
    !    !$acc declare device_resident(block_sum)
    !    NUM_BLOCKS = ceiling(real(size) / real(LOCAL_BUFFER_SIZE))
    !    allocate(block_sum(NUM_BLOCKS))
    !    call prefix_sum_full(vector, block_sum, NUM_BLOCKS, size)        
    !    if (NUM_BLOCKS > 1) then
    !        call prefix_sum_openacc(block_sum, int(NUM_BLOCKS, int32))            
    !        call add_block_sum(vector, block_sum, int(size, int32), NUM_BLOCKS,  LOCAL_BUFFER_SIZE)
    !    end if
    !    deallocate(block_sum)
    !end subroutine prefix_sum_openacc
