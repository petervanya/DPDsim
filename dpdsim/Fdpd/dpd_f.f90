module dpd_f

implicit none
contains


real(8) function randd()
    call random_number(randd)
end function


pure function wr(nr)
    real(8), intent(in) :: nr
    real(8) :: wr
    wr = 0.0
    if (nr <= 1.0) then
        wr = 1.0 - nr
    endif
end function


function gaussnum() result (y)
    real(8) :: x1, x2, w, y
    w = 2.0
    do while (w >= 1.0)
        x1 = 2.0 * randd() - 1.0
        x2 = 2.0 * randd() - 1.0
        w = x1**2 + x2**2
    enddo

    w = sqrt((-2.0 * log(w)) / w)
    y = x1 * w
end function


function compute_force_fun(X, V, bl, ip, box, gama, kT, dt) result (F)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), V(:, :), ip(:, :), box(3, 3)
    real(8), intent(in) :: gama, kT, dt
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), fpair, a, nr
    integer :: i, j, N
    real(8) :: fcube(size(X, 1), size(X, 1), 3), F(size(X, 1), 3)
    N = size(X, 1)
    fcube = 0.0
    g = 0.0
    inv_box = 0.0  ! do not forget this, else will spend hours debugging

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            a = ip(bl(i), bl(j))
            nr = norm2(rij)

            fpair = a * wr(nr) &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fcube(i, j, :) = fpair / nr * rij(:)
            fcube(j, i, :) = -fcube(i, j, :) !fpair / nr * rij(:)
        enddo
    enddo
    F = sum(fcube, 2)
end function


subroutine compute_force(F, vir, sigma, X, V, bl, ip, box, gama, kT, dt)
    real(8), intent(inout) :: F(:, :), vir, sigma(3)
    real(8), intent(in) :: X(:, :), V(:, :), ip(:, :), box(3, 3)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: gama, kT, dt
    !real(8), intent(out) :: F(size(X, 1), size(X, 2)), vir, sigma(3) ! DOES NOT WORK WITH F2PY!
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), volume, fpair, a, nr
    integer :: i, j, N
    real(8) :: fcube(size(X, 1), size(X, 1), 3)
    N = size(X, 1)
    fcube = 0.0
    g = 0.0
    inv_box = 0.0
    volume = 1.0

    vir = 0.0
    sigma = 0.0
    F = 0.0

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)
!    forall (i = 1:3) volume = volume * box(i, i) ! DOES NOT WORK CORRECTLY
    do i = 1, 3
        volume = volume * box(i, i)
    enddo

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            a = ip(bl(i), bl(j))
            nr = norm2(rij)

            fpair = a * wr(nr) &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fcube(i, j, :) = fpair / nr * rij(:)
            fcube(j, i, :) = -fcube(i, j, :) !fpair / nr * rij(:)

            vir = vir + dot_product(fcube(i, j, :), rij(:))
            sigma(:) = sigma(:) + fcube(i, j, :) * rij(:)
        enddo
    enddo

    do i = 1, N  ! kinetic term of the stress tensor
        sigma(:) = sigma(:) + V(i, :) * V(i, :)
    enddo

!    print *, "Virial = ", vir ! PASSES ZERO TO PYTHON!
    sigma = sigma / volume
    F = sum(fcube, 2)
end subroutine


function compute_force_cube(X, V, bl, ip, box, gama, kT, dt) result (fcube)
    real(8), intent(in) :: X(:, :), V(:, :), ip(:, :), box(3, 3)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: gama, kT, dt
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), fpair, a, nr
    integer :: i, j, N
    real(8) :: fcube(size(X, 1), size(X, 1), 3)
    N = size(X, 1)
    fcube = 0.0
    g = 0.0
    inv_box = 0.0

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            a = ip(bl(i), bl(j))
            nr = norm2(rij)

            fpair = a * wr(nr) &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fcube(i, j, :) = fpair / nr * rij(:)
            fcube(j, i, :) = -fcube(i, j, :) !fpair / nr * rij(:)
        enddo
    enddo
end function


function compute_pe(X, bl, ip, box) result (pe)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), ip(:, :), box(3, 3)
    integer :: i, j, N
    real(8) :: pe, rij(3), g(3), inv_box(3, 3)
    pe = 0.0
    N = size(X, 1)
    inv_box = 0.0  ! do not forget this, else will spend hours debugging

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, n
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            pe = pe + ip(bl(i), bl(j)) * wr(norm2(rij))**2 / 2.0
        enddo
    enddo
end function


function compute_ke(V) result (ke)
    real(8), intent(in) :: V(:, :)
    real(8) :: ke
    integer :: i, j, N
    ke = 0.0
    N = size(V, 1)

    do i = 1, N
        do j = 1, 3
            ke = ke + V(i, j) * V(i, j) / 2.0
        enddo
    enddo
end function


function init_pos(N, box) result(X)
    integer, intent(in) :: N
    real(8), intent(in) :: box(3, 3)
    real(8) :: X(N, 3)
    integer :: i, j

    do i = 1, N
        do j = 1, 3
            X(i, j) = randd() * box(j, j)
        enddo
    enddo
end function


function init_vel(N, kT) result(V)
    integer, intent(in) :: N
    real(8), intent(in) :: kT
    real(8) :: V(N, 3), Vcom(3)
    integer :: i, j

    do i = 1, N
        do j = 1, 3
            V(i, j) = gaussnum() * kT
        enddo
    enddo
    Vcom = sum(V, 1) / N
    do i = 1, N
        V(i, :) = V(i, :) - Vcom(:)
    enddo
end function


subroutine euler_step(X, V, bl, ip, box, gama, kT, dt)
    real(8), intent(inout) :: X(:, :), V(:, :)
    real(8), intent(in) :: ip(:, :), box(3, 3)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: gama, kT, dt
    real(8) :: F(size(X, 1), 3)
    F = 0.0

    F = compute_force_fun(X, V, bl, ip, box, gama, kT, dt)
    V(:, :) = V(:, :) + F(:, :) * dt
    X(:, :) = X(:, :) + V(:, :) * dt
end subroutine


subroutine verlet_step(X, V, F, bl, ip, box, gama, kT, dt)
    real(8), intent(inout) :: X(:, :), V(:, :), F(:, :)
    real(8), intent(in) :: ip(:, :), box(3, 3)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: gama, kT, dt
    real(8) :: V2(size(X, 1), 3)
    V2 = 0.0

    V2(:, :) = V(:, :) + 0.5 * F(:, :) * dt
    X(:, :) = X(:, :) + V2(:, :) * dt
    F = compute_force_fun(X, V2, bl, ip, box, gama, kT, dt)
    V(:, :) = V2(:, :) + 0.5 * F(:, :) * dt
end subroutine


subroutine print_mat(A)
    real(8), intent(in) :: A(:, :)
    integer :: M, i
    M = size(A, 1)
    do i = 1, M
        print '(f5.3, " ", f5.3, " ", f5.3)', A(i, 1), A(i, 2), A(i, 3)
    enddo
end subroutine


end module


