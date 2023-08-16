program dpd_sim

use dpd_f, only: init_pos, init_vel
use dpd_io, only: write_xyz, read_xyz
implicit none


integer :: N, Nt, Neq, thermo, df, i
integer, allocatable :: bl(:)
real(8) :: L, box(3, 3), ip(1, 1), gama, kT, dt
real(8) :: ti, tf
real(8), allocatable :: X(:, :), V(:, :), KE(:), PE(:)
integer :: si, sn
!character(30) :: infile
sn = 12
call random_seed(size=sn)
call random_seed(put=(/ (0, si=1, sn) /))


N = 375
L = 5.0
kT = 1.0
gama = 4.5
dt = 0.05
Nt = 300
Neq = 290
thermo = 10
df = 1
box = 0.0
forall (i = 1:3) box(i, i) = L
allocate(KE(N))
allocate(PE(N))
ip(1, 1) = 50.0

allocate(X(N, 3))
allocate(bl(N))
allocate(V(N, 3))
X = init_pos(N, box)
V = init_vel(N, kT)
bl = 1

!infile = "CONFIG.xyz"
!call read_xyz(infile, bl, X)
!print *, "Reading done."
!deallocate(bl)
!infile = "CONFIG.vel"
!call read_xyz(infile, bl, V)
!print *, "Velocities initialised."

call cpu_time(ti)
call integrate(X, V, size(X, 1), ip, 1, bl, box, &
    gama, kT, dt, Nt, Neq, thermo, df, KE, PE)
call cpu_time(tf)
print '("Time: ", f6.3, " s.")', tf - ti
end program


subroutine integrate(X, V, N, ip, Np, bl, box, &
        gama, kT, dt, Nt, Neq, thermo, df, KE, PE)
    use dpd_f, only: compute_pe, compute_ke, compute_force_fun, euler_step, verlet_step
    use dpd_io, only: write_xyz
    implicit none
    integer, intent(in) :: N, Np
    integer, intent(in) :: bl(N), Nt, Neq, thermo, df
    real(8), intent(in) :: ip(Np, Np), box(3, 3)
    real(8), intent(in) :: gama, kT, dt
    real(8), intent(inout) :: X(N, 3), V(N, 3)
    real(8), intent(out) :: KE(Nt+1), PE(Nt+1)
    real(8) :: T, F(size(X, 1), 3)
    integer :: it, i, j
    character(:), allocatable :: fname
    character(10) :: snum
    F = 0.0

    print '("dt: ",f5.3," | kT: ",f5.3," | gama:",f5.3)', dt, kT, gama

    F = compute_force_fun(X, V, bl, ip, box, gama, kT, dt)
!    ou = 20
!    open(unit=ou, file="init_force.out", action="write")
!    do i = 1, N
!        write(ou, *) F(i, :)
!    enddo
!    close(ou)

    KE(1) = compute_ke(V)
    PE(1) = compute_pe(X, bl, ip, box)
    print *, "Initial values:", KE(1), PE(1)

    print *, "Integrating..."
    do it = 2, Nt+1
!        call euler_step(X, V, bl, ip, box, gama, kT, dt)
        call verlet_step(X, V, F, bl, ip, box, gama, kT, dt)

        do i = 1, N
            do j = 1, 3
                X(i, j) = mod(X(i, j) + box(j, j), box(j, j))
            enddo
        enddo
        KE(it) = compute_ke(V)
        PE(it) = compute_pe(X, bl, ip, box)
        T = KE(it) / ((3 * N - 3) / 2.0)

        if (mod(it, thermo) == 0) then
            print *, it, T, KE(it), PE(it)
        endif
        if (mod(it, df) == 0 .and. it > Neq) then
            write(snum, "(i5.5)") it
            fname = "Dump/dump_" // trim(snum) // ".xyz"
            call write_xyz(fname, bl, X)
        endif
    enddo
end subroutine


