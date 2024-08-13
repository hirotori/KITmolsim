module py_interface_m
    !! interface module for interconnecting fortran and python via C-interface
    use iso_c_binding
    use distance_m
    use rdf_m
    implicit none
    
contains

subroutine calc_distance_pbc_interface(N, L, r, mat) bind(c, name="fort_calc_distance_pbc")
    integer(c_int),intent(in) :: N
    real(c_double),intent(in) :: L(3)
    real(c_double),intent(in) :: r(3,N)
    real(c_double),intent(inout) :: mat(N,N)

    call calc_distance_pbc(N, L, r, mat)

end subroutine


subroutine compute_radial_distribution_cinterf(n, natom, r, dr, Lbox, h) &
    bind(c, name="fort_compute_radial_distribution")
        !! compute radial distribution function
        integer(c_int32_t),intent(in) :: n
            !! number of histogram
        integer(c_int32_t),intent(in) :: natom
            !! number of atom
        real(c_double),intent(in) :: r(3,natom)
            !! position of atoms
        real(c_double),intent(in) :: dr
            !! width of bins
        real(c_double),intent(in) :: Lbox(3)
            !! box dimension
        real(c_double),intent(out) :: h(n)
            !! radial distribution
    
        call compute_radial_distribution(n, natom, r, dr, Lbox, h)
    
end subroutine
    
subroutine compute_radial_distribution_NP_cinterf(n, nsol, n_np, r_np, rsol, dr, Lbox, h) &
    bind(c, name="fort_compute_radial_distribution_NP")
        !! compute radial distribution function
        integer(c_int32_t),intent(in) :: n
            !! number of histogram
        integer(c_int32_t),intent(in) :: nsol
            !! number of atom
        integer(c_int32_t),intent(in) :: n_np
            !! number of NP
        real(c_double),intent(in) :: r_np(3,n_np)
            !! center-of-mass of NP
        real(c_double),intent(in) :: rsol(3,nsol)
            !! position of atoms
        real(c_double),intent(in) :: dr
            !! width of bins
        real(c_double),intent(in) :: Lbox(3)
            !! box dimension
        real(c_double),intent(out) :: h(n)
            !! radial distribution
    
        call compute_radial_distribution_NP(n, nsol, n_np, r_np, rsol, dr, Lbox, h)
    
end subroutine
    

end module py_interface_m