module dpd_io

implicit none

contains

subroutine write_xyz(fname, bl, X)
    character(len=20), intent(in) :: fname
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :)
    integer :: N, i, ou
    N = size(X, 1)
    ou=20

    open(unit=ou, file=fname, action="write", status="replace")
    write(ou, *) N
    do i = 1, N
        write(ou, *) bl(i), X(i, :)
    enddo
    close(ou)
end subroutine

end module
