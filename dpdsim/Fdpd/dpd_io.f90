module dpd_io

implicit none

contains

subroutine write_xyz(fname, bl, X)
    character(*), intent(in) :: fname
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :)
    integer :: N, i, ou
    N = size(X, 1)
    ou=20

    open(unit=ou, file=fname, action="write")
    write(ou, *) N
    write(ou, *) "bla"
    do i = 1, N
        write(ou, *) bl(i), X(i, :)
    enddo
    close(ou)
end subroutine


subroutine read_xyz(fname, bl, X)
    character(30), intent(in) :: fname
    integer, allocatable :: bl(:)
    real(8), allocatable :: X(:, :)
    integer :: N, i, ou
    character(10) :: blah
    ou=20 

    open(unit=ou, file=fname, action="read")
    read(ou, *) N
    read(ou, *) blah
    print '("Reading", i5, " beads...")', N
    allocate(X(N, 3))
    allocate(bl(N))

    do i = 1, N
        read(ou, *) bl(i), X(i, :)
    enddo
    close(ou)

!    do i = n-10, n
!        print *, X(i, :) 
!    enddo
end subroutine

end module
