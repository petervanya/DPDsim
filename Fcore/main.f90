program dpd_sim

use dpd_f, only: init_pos, init_vel
use dpd_io, only: write_xyz
implicit none


integer :: N, Nt, thermo
integer :: i, j
integer, allocatable :: blist(:)
real(8) :: sp(1, 1), L, box(3, 3), ip(1, 1), gama, kT, dt
real(8), allocatable :: X(:, :), V(:, :), KE(:), PE(:)

N = 400
L = 5.0
thermo = 1
kT = 1.0
gama = 4.5
dt = 0.002
Nt = 10
allocate(X(N, 3))
allocate(V(N, 3))
allocate(blist(N))
forall (i = 1:3) box(i, i) = L
allocate(KE(N))
allocate(PE(N))
blist = 1
ip(1, 1) = 25.0

X = init_pos(N, box)
V = init_vel(N, kT)

!do i = 1, N
!    print *, V(i, :) 
!enddo

call integrate_euler(X, V, N, ip, 1, blist, box, &
    gama, kT, dt, Nt, thermo, KE, PE)

end program


subroutine integrate_euler(X, V, N, ip, Np, blist, box, &
        gama, kT, dt, Nt, thermo, KE, PE)
    use dpd_f, only: tot_pe, tot_ke, force_list
    use dpd_io, only: write_xyz
    implicit none
    integer, intent(in) :: N, Np
    integer, intent(in) :: blist(N), Nt, thermo
    real(8), intent(in) :: ip(Np, Np), box(3, 3)
    real(8), intent(in) :: gama, kT, dt
    real(8), intent(inout) :: X(N, 3), V(N, 3)
    real(8), intent(out) :: KE(Nt), PE(Nt)
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), T
    real(8) :: F(size(X, 1), 3)
    integer :: it, i, j
    character(:), allocatable :: fname
    character(10) :: snum

    do it = 1, Nt
        F = force_list(X, V, ip, blist, box, gama, kT, dt)
        do i = 1, N
            V(i, :) = V(i, :) + F(i, :) * dt
            X(i, :) = X(i, :) + V(i, :) * dt
        enddo
        do i = 1, N
            do j = 1, 3
                X(i, j) = mod(X(i, j) + box(j, j), box(j, j))
            enddo
        enddo
        PE(it) = tot_pe(X, ip, blist, box)
        KE(it) = tot_ke(V)
        T = KE(it) / (3.0 / 2.0 * N)

        if (mod(it, thermo) == 0) then
            write(snum, "(i5.5)") it
            fname = "Dump/dump_" // trim(snum) // ".xyz"
            call write_xyz(fname, blist, X)
            print *, it, T, KE(it), PE(it)
        endif
    enddo
end subroutine


!subroutine integrate_verlet(X, V, ip, blist, box, &
!        gama, kT, dt, nt)
!    integer, intent(in) :: n, np, blist(n), nt
!    real(8), intent(in) :: X(n, 3), V(n, 3), ip(np, np), box(3, 3)
!    real(8), intent(in) :: gama, kT, dt
!    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3)
!end subroutine


