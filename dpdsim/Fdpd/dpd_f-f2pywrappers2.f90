!     -*- f90 -*-
!     This file is autogenerated with f2py (version:2)
!     It contains Fortran 90 wrappers to fortran functions.

      subroutine f2pywrap_dpd_f_randd (randdf2pywrap)
      use dpd_f, only : randd
      real(kind=8) randdf2pywrap
      randdf2pywrap = randd()
      end subroutine f2pywrap_dpd_f_randd
      subroutine f2pywrap_dpd_f_wr (wrf2pywrap, nr)
      use dpd_f, only : wr
      real(kind=8) nr
      real(kind=8) wrf2pywrap
      wrf2pywrap = wr(nr)
      end subroutine f2pywrap_dpd_f_wr
      subroutine f2pywrap_dpd_f_gaussnum (gaussnumf2pywrap)
      use dpd_f, only : gaussnum
      real(kind=8) gaussnumf2pywrap
      gaussnumf2pywrap = gaussnum()
      end subroutine f2pywrap_dpd_f_gaussnum
      subroutine f2pywrap_dpd_f_compute_force_fun (compute_force_funf2py&
     &wrap, X, V, bl, ip, box, gama, kT, dt, f2py_X_d0, f2py_X_d1, f2py_&
     &V_d0, f2py_V_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      use dpd_f, only : compute_force_fun
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      real(kind=8) compute_force_funf2pywrap(size(X, 1),3)
      compute_force_funf2pywrap = compute_force_fun(X, V, bl, ip, box, g&
     &ama, kT, dt)
      end subroutine f2pywrap_dpd_f_compute_force_fun
      subroutine f2pywrap_dpd_f_compute_force (F, vir, sigma, X, V, bl, &
     &ip, box, gama, kT, dt, f2py_F_d0, f2py_F_d1, f2py_X_d0, f2py_X_d1,&
     & f2py_V_d0, f2py_V_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      use dpd_f, only : compute_force
      real(kind=8) vir
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_F_d0
      integer f2py_F_d1
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) F(f2py_F_d0,f2py_F_d1)
      real(kind=8) sigma(3)
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      call compute_force(F, vir, sigma, X, V, bl, ip, box, gama, kT, dt)
      end subroutine f2pywrap_dpd_f_compute_force
      subroutine f2pywrap_dpd_f_compute_force_cube (compute_force_cubef2&
     &pywrap, X, V, bl, ip, box, gama, kT, dt, f2py_X_d0, f2py_X_d1, f2p&
     &y_V_d0, f2py_V_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      use dpd_f, only : compute_force_cube
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      real(kind=8) compute_force_cubef2pywrap(size(X, 1),size(X, 1),3)
      compute_force_cubef2pywrap = compute_force_cube(X, V, bl, ip, box,&
     & gama, kT, dt)
      end subroutine f2pywrap_dpd_f_compute_force_cube
      subroutine f2pywrap_dpd_f_compute_pe (compute_pef2pywrap, X, bl, i&
     &p, box, f2py_X_d0, f2py_X_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      use dpd_f, only : compute_pe
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      real(kind=8) compute_pef2pywrap
      compute_pef2pywrap = compute_pe(X, bl, ip, box)
      end subroutine f2pywrap_dpd_f_compute_pe
      subroutine f2pywrap_dpd_f_compute_ke (compute_kef2pywrap, V, f2py_&
     &V_d0, f2py_V_d1)
      use dpd_f, only : compute_ke
      integer f2py_V_d0
      integer f2py_V_d1
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      real(kind=8) compute_kef2pywrap
      compute_kef2pywrap = compute_ke(V)
      end subroutine f2pywrap_dpd_f_compute_ke
      subroutine f2pywrap_dpd_f_init_pos (init_posf2pywrap, N, box)
      use dpd_f, only : init_pos
      integer N
      real(kind=8) box(3,3)
      real(kind=8) init_posf2pywrap(N,3)
      init_posf2pywrap = init_pos(N, box)
      end subroutine f2pywrap_dpd_f_init_pos
      subroutine f2pywrap_dpd_f_init_vel (init_velf2pywrap, N, kT)
      use dpd_f, only : init_vel
      integer N
      real(kind=8) kT
      real(kind=8) init_velf2pywrap(N,3)
      init_velf2pywrap = init_vel(N, kT)
      end subroutine f2pywrap_dpd_f_init_vel
      subroutine f2pywrap_dpd_f_euler_step (X, V, bl, ip, box, gama, kT,&
     & dt, f2py_X_d0, f2py_X_d1, f2py_V_d0, f2py_V_d1, f2py_bl_d0, f2py_&
     &ip_d0, f2py_ip_d1)
      use dpd_f, only : euler_step
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      call euler_step(X, V, bl, ip, box, gama, kT, dt)
      end subroutine f2pywrap_dpd_f_euler_step
      subroutine f2pywrap_dpd_f_verlet_step (X, V, F, bl, ip, box, gama,&
     & kT, dt, f2py_X_d0, f2py_X_d1, f2py_V_d0, f2py_V_d1, f2py_F_d0, f2&
     &py_F_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      use dpd_f, only : verlet_step
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_F_d0
      integer f2py_F_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      real(kind=8) F(f2py_F_d0,f2py_F_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      call verlet_step(X, V, F, bl, ip, box, gama, kT, dt)
      end subroutine f2pywrap_dpd_f_verlet_step
      subroutine f2pywrap_dpd_f_print_mat (A, f2py_A_d0, f2py_A_d1)
      use dpd_f, only : print_mat
      integer f2py_A_d0
      integer f2py_A_d1
      real(kind=8) A(f2py_A_d0,f2py_A_d1)
      call print_mat(A)
      end subroutine f2pywrap_dpd_f_print_mat
      
      subroutine f2pyinitdpd_f(f2pysetupfunc)
      interface 
      subroutine f2pywrap_dpd_f_randd (randdf2pywrap, randd)
      real(kind=8) randd
      real(kind=8) randdf2pywrap
      end subroutine f2pywrap_dpd_f_randd 
      subroutine f2pywrap_dpd_f_wr (wrf2pywrap, wr, nr)
      real(kind=8) wr
      real(kind=8) nr
      real(kind=8) wrf2pywrap
      end subroutine f2pywrap_dpd_f_wr 
      subroutine f2pywrap_dpd_f_gaussnum (gaussnumf2pywrap, gaussnum)
      real(kind=8) gaussnum
      real(kind=8) gaussnumf2pywrap
      end subroutine f2pywrap_dpd_f_gaussnum 
      subroutine f2pywrap_dpd_f_compute_force_fun (compute_force_funf2py&
     &wrap, compute_force_fun, X, V, bl, ip, box, gama, kT, dt, f2py_X_d&
     &0, f2py_X_d1, f2py_V_d0, f2py_V_d1, f2py_bl_d0, f2py_ip_d0, f2py_i&
     &p_d1)
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      real(kind=8) compute_force_fun(size(X, 1),3)
      real(kind=8) compute_force_funf2pywrap(size(X, 1),3)
      end subroutine f2pywrap_dpd_f_compute_force_fun 
      subroutine f2pywrap_dpd_f_compute_force (F, vir, sigma, X, V, bl, &
     &ip, box, gama, kT, dt, f2py_F_d0, f2py_F_d1, f2py_X_d0, f2py_X_d1,&
     & f2py_V_d0, f2py_V_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      real(kind=8) vir
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_F_d0
      integer f2py_F_d1
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) F(f2py_F_d0,f2py_F_d1)
      real(kind=8) sigma(3)
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      end subroutine f2pywrap_dpd_f_compute_force 
      subroutine f2pywrap_dpd_f_compute_force_cube (compute_force_cubef2&
     &pywrap, compute_force_cube, X, V, bl, ip, box, gama, kT, dt, f2py_&
     &X_d0, f2py_X_d1, f2py_V_d0, f2py_V_d1, f2py_bl_d0, f2py_ip_d0, f2p&
     &y_ip_d1)
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      real(kind=8) compute_force_cube(size(X, 1),size(X, 1),3)
      real(kind=8) compute_force_cubef2pywrap(size(X, 1),size(X, 1),3)
      end subroutine f2pywrap_dpd_f_compute_force_cube 
      subroutine f2pywrap_dpd_f_compute_pe (compute_pef2pywrap, compute_&
     &pe, X, bl, ip, box, f2py_X_d0, f2py_X_d1, f2py_bl_d0, f2py_ip_d0, &
     &f2py_ip_d1)
      real(kind=8) compute_pe
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      real(kind=8) compute_pef2pywrap
      end subroutine f2pywrap_dpd_f_compute_pe 
      subroutine f2pywrap_dpd_f_compute_ke (compute_kef2pywrap, compute_&
     &ke, V, f2py_V_d0, f2py_V_d1)
      real(kind=8) compute_ke
      integer f2py_V_d0
      integer f2py_V_d1
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      real(kind=8) compute_kef2pywrap
      end subroutine f2pywrap_dpd_f_compute_ke 
      subroutine f2pywrap_dpd_f_init_pos (init_posf2pywrap, init_pos, N,&
     & box)
      integer N
      real(kind=8) box(3,3)
      real(kind=8) init_pos(N,3)
      real(kind=8) init_posf2pywrap(N,3)
      end subroutine f2pywrap_dpd_f_init_pos 
      subroutine f2pywrap_dpd_f_init_vel (init_velf2pywrap, init_vel, N,&
     & kT)
      integer N
      real(kind=8) kT
      real(kind=8) init_vel(N,3)
      real(kind=8) init_velf2pywrap(N,3)
      end subroutine f2pywrap_dpd_f_init_vel 
      subroutine f2pywrap_dpd_f_euler_step (X, V, bl, ip, box, gama, kT,&
     & dt, f2py_X_d0, f2py_X_d1, f2py_V_d0, f2py_V_d1, f2py_bl_d0, f2py_&
     &ip_d0, f2py_ip_d1)
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      end subroutine f2pywrap_dpd_f_euler_step 
      subroutine f2pywrap_dpd_f_verlet_step (X, V, F, bl, ip, box, gama,&
     & kT, dt, f2py_X_d0, f2py_X_d1, f2py_V_d0, f2py_V_d1, f2py_F_d0, f2&
     &py_F_d1, f2py_bl_d0, f2py_ip_d0, f2py_ip_d1)
      real(kind=8) gama
      real(kind=8) kT
      real(kind=8) dt
      integer f2py_X_d0
      integer f2py_X_d1
      integer f2py_V_d0
      integer f2py_V_d1
      integer f2py_F_d0
      integer f2py_F_d1
      integer f2py_bl_d0
      integer f2py_ip_d0
      integer f2py_ip_d1
      real(kind=8) X(f2py_X_d0,f2py_X_d1)
      real(kind=8) V(f2py_V_d0,f2py_V_d1)
      real(kind=8) F(f2py_F_d0,f2py_F_d1)
      integer bl(f2py_bl_d0)
      real(kind=8) ip(f2py_ip_d0,f2py_ip_d1)
      real(kind=8) box(3,3)
      end subroutine f2pywrap_dpd_f_verlet_step 
      subroutine f2pywrap_dpd_f_print_mat (A, f2py_A_d0, f2py_A_d1)
      integer f2py_A_d0
      integer f2py_A_d1
      real(kind=8) A(f2py_A_d0,f2py_A_d1)
      end subroutine f2pywrap_dpd_f_print_mat
      end interface
      external f2pysetupfunc
      call f2pysetupfunc(f2pywrap_dpd_f_randd,f2pywrap_dpd_f_wr,f2pywrap&
     &_dpd_f_gaussnum,f2pywrap_dpd_f_compute_force_fun,f2pywrap_dpd_f_co&
     &mpute_force,f2pywrap_dpd_f_compute_force_cube,f2pywrap_dpd_f_compu&
     &te_pe,f2pywrap_dpd_f_compute_ke,f2pywrap_dpd_f_init_pos,f2pywrap_d&
     &pd_f_init_vel,f2pywrap_dpd_f_euler_step,f2pywrap_dpd_f_verlet_step&
     &,f2pywrap_dpd_f_print_mat)
      end subroutine f2pyinitdpd_f


