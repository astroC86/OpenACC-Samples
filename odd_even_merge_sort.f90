
module array_checker
    use iso_fortran_env
    implicit none

    private
    public :: check_arrays

contains
    subroutine count_frequencies(arr, n, freq, max_val)
        integer, intent(in) :: n, max_val
        integer, intent(in) :: arr(n)
        integer, intent(out) :: freq(0:max_val)
        integer :: i
        
        freq = 0
        do i = 1, n
            freq(arr(i)) = freq(arr(i)) + 1
        end do
    end subroutine

    subroutine check_arrays(input_arr, n)
        integer, intent(in) :: n
        integer, intent(in) :: input_arr(n)
        integer :: i, j, temp, difference_count
        logical :: differences_found
        
        write(*, '(A)') "Checking array equivalence..."
        write(*, '(A,I0)') "Length of arrays: ", n
  
        
        differences_found = .false.
        difference_count  = 0
        do i = 1, n
            if (input_arr(i) /= i) then
                difference_count =  difference_count + 1
                if (.not. differences_found) then
                    write(*, '(A)') "(X) Differences found:"
                    differences_found = .true.
                end if
                write(*, '(A,I0,A,I0)') "Expected Number ", i, " but got ", input_arr(i)
                if (difference_count >= 5) exit
            end if
        end do
        
        if (.not. differences_found) then
            write(*, '(A)') "(✓) Arrays are equivalent when sorted!"
        end if
        
    end subroutine
end module

!Execution time: 3191377.80 microseconds 512
!Execution time: 1811619.90 microseconds 1024
!Execution time: 1083942.10 microseconds 2048
!Execution time:  474055.00 microseconds 4096
!Execution time:  314867.90 microseconds 4096*2
!Execution time:  175388.00 microseconds 4096*2*2
!Execution time:  144398.90 microseconds 4096*2*2*2
!Execution time:  181143.90 microseconds 4096*2*2*2*2
module sorter
    implicit none 
    !integer, parameter :: N = 1024
    integer, parameter :: SHARED_SIZE_LIMIT = 65536 !4096*2*2*2
    integer, parameter :: THREADS_PER_BLOCK = SHARED_SIZE_LIMIT/2
    integer, parameter :: dir = 1

    contains

    subroutine get_time_us(time)
        real(kind=8), intent(out) :: time
        integer(kind=8) :: count, count_rate
        
        call system_clock(count, count_rate)
        time = real(count, 8) * 1000000.0d0 / real(count_rate, 8)
    end subroutine

    subroutine print_array(arr, label)
        integer, intent(in) :: arr(:)
        character(len=*), intent(in) :: label
        integer :: i
        
        write(*, '(A)') trim(label)
        do i = 1, min(size(arr), 1024)  ! Print all elements or first 125 if array is larger
            write(*, '(I4)', advance='no') arr(i)
            if (mod(i, 25) == 0) write(*,*)  ! New line every 25 elements
        end do
        write(*,*)  ! Final newline
    end subroutine

    pure function log2(x) result(l)
        real, intent(in) :: x
        real :: l
        l = real(floor(real(log(real(x)) / log(2.0))))
    end function log2


    subroutine odd_even_merge_sort_global(NUM_BLOCKS, NUM_THREADS, d_DstKey,d_DstVal, d_SrcKey, d_SrcVal, size, stride)
        use iso_fortran_env
        implicit none

        integer(int32) , intent(inout) :: d_DstKey(:), d_DstVal(:)
        integer(int32) , intent(inout) :: d_SrcKey(:), d_SrcVal(:)
        integer(int32) , intent(in)    :: size
        integer(int32) , intent(in)    :: stride
        integer(int32) , intent(in)    :: NUM_BLOCKS, NUM_THREADS

        integer(int32)                 :: tid, bid, global_comparatorI, temp
        integer(int32)                 :: offset, pos, keyA,valA,keyB,valB

        !$acc parallel loop num_gangs(num_blocks) vector_length(NUM_THREADS) &
        !$acc& present(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal)
        do bid = 1, NUM_BLOCKS
            !$acc loop vector
            do tid = 1, NUM_THREADS                
                global_comparatorI = (bid - 1) * NUM_THREADS + (tid - 1)  ! Corrected
                pos = 2 * global_comparatorI - iand(global_comparatorI, (stride - 1)) + 1
                if (stride < size/2) then
                    offset = iand(global_comparatorI, (size/2) - 1)
                    if (offset >= stride) then
                        keyA = d_SrcKey(pos - stride);
                        valA = d_SrcVal(pos - stride);
                        
                        keyB = d_SrcKey(pos +      0);
                        valB = d_SrcVal(pos +      0);

                        if ((keyA > keyB) .and. (dir ==1)) then
                            temp = keyA
                            keyA = keyB
                            keyB = temp

                            temp = valA
                            valA = valB
                            valB = temp
                        end if                        
                                    
                        d_DstKey(pos - stride) = keyA;
                        d_DstVal(pos - stride) = valA;

                        d_DstKey(pos +      0) = keyB;
                        d_DstVal(pos +      0) = valB;
                    end if
                else
                    keyA = d_SrcKey(pos +      0);
                    valA = d_SrcVal(pos +      0);

                    keyB = d_SrcKey(pos + stride);
                    valB = d_SrcVal(pos + stride);

                    if ((keyA > keyB) .and. (dir ==1)) then
                        temp = keyA
                        keyA = keyB
                        keyB = temp
                        
                        temp = valA
                        valA = valB
                        valB = temp
                    end if

                    d_DstKey(pos +      0) = keyA;
                    d_DstVal(pos +      0) = valA;

                    d_DstKey(pos + stride) = keyB;
                    d_DstVal(pos + stride) = valB;
                end if
            end do 
        end do
        !$acc end parallel loop
    end subroutine

    subroutine odd_even_merge_sort_shared(d_dst_key,d_dst_val,d_src_key,d_src_val, N, NUM_BLOCKS)
        use cudafor
        use array_checker
        integer,  intent(in)    :: N, NUM_BLOCKS
        integer , intent(inout) :: d_dst_key(:)
        integer , intent(inout) :: d_dst_val(:)
        integer , intent(in)    :: d_src_key(:)
        integer , intent(in)    :: d_src_val(:)

        integer :: s_key(SHARED_SIZE_LIMIT)
        integer :: s_val(SHARED_SIZE_LIMIT)
        integer :: block_idx,thread_idx, global_idx
        integer :: size, pos, stride, offset
        integer :: key_a, key_b, val_a, val_b

        !$acc parallel loop async num_gangs(num_blocks) vector_length(THREADS_PER_BLOCK) &
        !$acc& private(s_key,s_val) present(d_dst_key,d_dst_val,d_src_key,d_src_val)
        do block_idx = 1, num_blocks
            
            s_key(:) = 0

            !$acc loop vector private(global_idx)
            do thread_idx = 1, THREADS_PER_BLOCK
                global_idx = (block_idx - 1) * SHARED_SIZE_LIMIT + thread_idx
                if (global_idx <= N) then
                    s_key(thread_idx) = d_src_key(global_idx)
                    s_val(thread_idx) = d_src_val(global_idx)
                    if (global_idx + (SHARED_SIZE_LIMIT/2) <= N) then
                        s_key(thread_idx + SHARED_SIZE_LIMIT/2) = d_src_key(global_idx + (SHARED_SIZE_LIMIT/2))
                        s_val(thread_idx + SHARED_SIZE_LIMIT/2) = d_src_val(global_idx + (SHARED_SIZE_LIMIT/2))
                    else
                        s_key(thread_idx) = huge(0)
                        s_val(thread_idx) = 0
                    end if
                end if
            end do

            size = 2
            do while (size <= N)
                !$acc loop vector private(key_b,pos,key_a,val_b,val_a)
                do thread_idx = 1, THREADS_PER_BLOCK
                    pos = 2 * (thread_idx - 1) - mod(thread_idx - 1, size/2) + 1
                    key_a = s_key(pos)
                    val_a = s_val(pos)
                    key_b = s_key(pos + size/2)
                    val_b = s_val(pos + size/2)
                    
                    if ( (key_a > key_b) .eqv. (dir == 1)) then
                        s_key(pos) = key_b
                        s_val(pos) = val_b
                        s_key(pos + size/2) = key_a
                        s_val(pos + size/2) = val_a
                    end if
                end do
                
                stride = size/4
                do while (stride > 0)
                    !$acc loop vector private(pos, offset, key_a, key_b, val_a, val_b)
                    do thread_idx = 1, THREADS_PER_BLOCK
                        pos = 2 *  (thread_idx - 1) - mod(thread_idx - 1, stride) + 1
                        offset = iand(thread_idx - 1, size/2 - 1) 
                        
                        if (offset >= stride) then
                            key_a = s_key(pos - stride)
                            val_a = s_val(pos - stride)
                            key_b = s_key(pos)
                            val_b = s_val(pos)
                            
                            if ((key_a > key_b) .eqv. (dir == 1)) then
                                s_key(pos - stride) = key_b
                                s_val(pos - stride) = val_b
                                s_key(pos) = key_a
                                s_val(pos) = val_a
                            end if
                        end if
                    end do
                    !$acc end loop
                    stride = ishft(stride, -1)
                end do
                size = size * 2
            end do
            
            !$acc loop vector private(global_idx)
            do thread_idx = 1, THREADS_PER_BLOCK
                global_idx = (block_idx - 1) * SHARED_SIZE_LIMIT + thread_idx
                if (global_idx <= N) then
                    d_dst_key(global_idx) = s_key(thread_idx)
                    d_dst_val(global_idx) = s_val(thread_idx)

                    if (global_idx + (SHARED_SIZE_LIMIT/2) <= N) then
                        d_dst_key(global_idx + (SHARED_SIZE_LIMIT/2)) = s_key(thread_idx + (SHARED_SIZE_LIMIT/2))
                        d_dst_val(global_idx + (SHARED_SIZE_LIMIT/2)) = s_val(thread_idx + (SHARED_SIZE_LIMIT/2))
                    end if

                end if
            end do
        end do
        !$acc end parallel
    end subroutine

    subroutine odd_even_merge_sort(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength)
        use openacc
        use iso_c_binding, only: c_associated, c_loc, c_ptr

        integer, allocatable, target, intent(inout) :: d_DstKey(:)
        integer, allocatable, target, intent(inout) :: d_DstVal(:)
        integer, allocatable, target, intent(in)    :: d_SrcKey(:)
        integer, allocatable, target, intent(in)    :: d_SrcVal(:)
        integer, intent(in)    :: arrayLength
        
        integer :: num_blocks
        integer :: log2L
        integer :: size, stride
        integer :: i
        logical :: direction
        
        if (arrayLength < 2) return
        
        log2L = floor(log2(real(arrayLength)))
        if (2**log2L /= arrayLength) then
            print *, "Error: Array length must be power of 2"
            return
        endif

        num_blocks = arrayLength / SHARED_SIZE_LIMIT;
        if (arrayLength <= SHARED_SIZE_LIMIT) then
            call odd_even_merge_sort_shared(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal,  arrayLength, num_blocks)
            !$acc wait
        else
            do i = 1, arrayLength, SHARED_SIZE_LIMIT
                call odd_even_merge_sort_shared(d_DstKey(i : min(i + SHARED_SIZE_LIMIT - 1, arrayLength)), &
                                                d_DstVal(i : min(i + SHARED_SIZE_LIMIT - 1, arrayLength)), & 
                                                d_SrcKey(i : min(i + SHARED_SIZE_LIMIT - 1, arrayLength)), &
                                                d_SrcVal(i : min(i + SHARED_SIZE_LIMIT - 1, arrayLength)), & 
                                                SHARED_SIZE_LIMIT, num_blocks)
            end do
            !$acc wait 
            size = 2 * SHARED_SIZE_LIMIT
            do while (size <= arrayLength)
                stride = size / 2
                do while (stride > 0)
                    call odd_even_merge_sort_global(arrayLength/512, 256, d_DstKey, d_DstVal, d_DstKey, d_DstVal, size, stride)
                    stride = stride / 2
                end do
                size = size * 2
            end do
        endif
    end subroutine odd_even_merge_sort

end module sorter

program test
    use sorter
    use array_checker
    integer              :: N          = 8388608!4194304
    integer              :: num_values = 1024
    integer, allocatable :: d_src_key(:), d_src_val(:)
    integer, allocatable :: d_dst_key(:), d_dst_val(:)
    integer, allocatable :: input_copy(:)

    integer ::  i
    real(kind=8) :: start_time, end_time
    integer :: seed_size
    integer,allocatable :: seed(:)

    call random_seed(size=seed_size)   ! n is processor-dependent
    allocate(seed(seed_size))
    seed = 42
    call random_seed(put=seed)

    allocate(d_src_key(N),d_src_val(N),d_dst_key(N),d_dst_val(N), input_copy(N))

    do i = 1, N
        d_src_key(i) = i
        d_src_val(i) = i

        d_dst_key(i) = -1
        d_dst_val(i) = -1
    end do

    call Shuffle(d_src_key)
    
    input_copy = d_src_key
    !call print_array(d_src_key, "Input Array:")
    !$acc data copy(d_dst_key, d_dst_val, d_src_key, d_src_val)
    call odd_even_merge_sort(d_dst_key, d_dst_val, d_src_key, d_src_val,N)

    call get_time_us(start_time)    
    call odd_even_merge_sort(d_dst_key, d_dst_val, d_src_key, d_src_val,N)
    call get_time_us(end_time)
    !$acc end data
    write(*, '(A,F10.2,A)') "Execution time: ", end_time - start_time, " microseconds"
    call check_arrays(d_dst_key, N)

    !call print_array(d_dst_key, "Sorted Array:")


    !call print_array(d_dst_key, "Sorted Array:")
    !call check_arrays(input_copy, d_dst_key, N, num_values-1)
    deallocate(d_src_key,d_src_val,d_dst_key,d_dst_val, input_copy)
    !call odd_even_merge_sort_shared()
    !call odd_even_merge_sort_shared()
    !call odd_even_merge_sort_shared()
    !call odd_even_merge_sort_shared()
    contains
        subroutine Shuffle(a)
            integer, intent(inout) :: a(:)
            integer :: i, randpos, temp
            real :: r
            do i = size(a), 2, -1
            call random_number(r)
            randpos = int(r * i) + 1
            temp = a(randpos)
            a(randpos) = a(i)
            a(i) = temp
            end do
        end subroutine Shuffle
end program test
