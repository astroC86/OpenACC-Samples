program laplace
  use openacc
  use nvtx

  implicit none
  integer, parameter :: fp_kind = kind(1.0d0)
  integer, parameter :: n = 2048, m = 2048, iter_max = 100
  integer :: i, j, iter, jb
  real(fp_kind), dimension(:,:,:), allocatable :: A 
  real(fp_kind), dimension(:)    , allocatable :: errors

  real(fp_kind) :: tol = 1.0e-6_fp_kind, error = 1.0_fp_kind, error_local = 1.0_fp_kind
  real(fp_kind) :: start_time, stop_time
  integer :: cur, nxt, counted, num_gangs, gang_id
  logical :: all_ready = .false.
  !$acc declare device_resident(errors)
  num_gangs =  512
  allocate( A(0:n-1, 0:m-1, 0:2) )
  allocate(errors(num_gangs))

  A(:,:,0) = 0.0_fp_kind
  A(:,:,1) = 0.0_fp_kind
  ! Set B.C.
  A(0,:,0) = 1.0_fp_kind
  A(0,:,1) = 1.0_fp_kind

  iter = 0
  !$acc data copyin(A)
  call nvtxRangePushA("while")
  do while ( error .gt. tol .and. iter .lt. iter_max )
    error       = 0.0_fp_kind
    
    !$acc kernels
    errors(:) = -1.0_fp_kind
    !$acc end kernels

    ! Determine which plane is “current” and which is “next”
    cur = mod(iter, 2)      ! 0 or 1
    nxt = 1 - cur           ! if cur=0, nxt=1; if cur=1, nxt=0

    call nvtxStartRange("calcNext ", 1)
    !$acc parallel loop gang present(A,errors) private(error_local) num_gangs(num_gangs) vector_length(1024)  async(1)
    do gang_id = 1, num_gangs
      error_local = 0.0_fp_kind
      !$acc loop seq
      do j = gang_id, m-2, num_gangs
        !$acc loop vector reduction(max: error_local)
        do i = 1, n-2
          A(i, j, nxt) = 0.25_fp_kind * ( A(i+1, j,   cur) + A(i-1, j,   cur) + &
                                          A(i,   j-1, cur) + A(i,   j+1, cur) )
          error_local = max( error_local, abs(A(i, j, nxt) - A(i, j, cur)) )
        end do
        !$acc end loop
      end do
      errors(gang_id) = error_local
    end do    
    !$acc end parallel loop
    
    !$acc parallel loop private(error_local, counted,all_ready) present(errors) copyout(error) num_gangs(1) vector_length(32) 
    do i=1,1
      counted     = 0
      error_local = 0.0_fp_kind

      do while (counted < num_gangs)
        all_ready = .true.
        !$acc loop vector reduction(.and.:all_ready)
        do j=1, 32 
          if (errors(counted + j) .ne. -1.0) then
            all_ready = .true.
          else 
            all_ready = .false.
          end if
        end do 

        if (all_ready) then
          !$acc loop vector reduction(max:error_local)
          do j=1, 32
            error_local = max(errors(counted + j), error_local)
          end do
          counted = counted + 32
        end if
      end do
      error = error_local
    end do
    !$acc end parallel loop
    !$acc wait(1)

    call nvtxEndRange()
    if(mod(iter,10).eq.0 ) write(*,'(i5,f10.6)'), iter, error
    iter = iter + 1
  end do
  call nvtxRangePop()
  !$acc end data
  deallocate(A)
  deallocate(errors)
end program laplace
