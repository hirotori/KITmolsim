module rdf_m
    implicit none
    private

    public compute_radial_distribution, compute_radial_distribution_NP, compute_rdf

contains
    subroutine compute_radial_distribution(n, natom, r, dr, Lbox, h)
        !! compute radial distribution function
        integer(4),intent(in) :: n
            !! number of histogram
        integer(4),intent(in) :: natom
            !! number of atom
        real(8),intent(in) :: r(3,natom)
            !! position of atoms
        real(8),intent(in) :: dr
            !! width of bins
        real(8),intent(in) :: Lbox(3)
            !! box dimension
        real(8),intent(out) :: h(n)
            !! radial distribution

        integer i, j, k
        real(8),dimension(3) :: ri, rij, iL
        real(8) dijsq

        if ( any(n*dr > Lbox*0.5d0) ) then
            error stop "Error::compute_rdf(rdf.f90):: bin width greater tham half length of the box"
        end if

        iL = 1.0/Lbox
        do i = 1, natom-1
            ri(:) = r(:,i)
            do j = i+1, natom
                rij(:) = ri - r(:,j)
                ! minimum image convention
                rij(:) = rij(:) - nint(rij(:)*iL(:))*Lbox(:)
                dijsq = sum(rij**2)
                k = floor(sqrt(dijsq)/dr) + 1
                if ( k <= n ) h(k) = h(k) + 2 ! Two distances of atoms i-j and j-i are simultaneously considered.
            end do            
        end do                

    end subroutine

    subroutine compute_radial_distribution_NP(n, nsol, n_np, r_np, rsol, dr, Lbox, h)
        !! compute radial distribution function
        integer(4),intent(in) :: n
            !! number of histogram
        integer(4),intent(in) :: nsol
            !! number of atom
        integer(4),intent(in) :: n_np
            !! number of NP
        real(8),intent(in) :: r_np(3,n_np)
            !! center-of-mass of NP
        real(8),intent(in) :: rsol(3,nsol)
            !! position of atoms
        real(8),intent(in) :: dr
            !! width of bins
        real(8),intent(in) :: Lbox(3)
            !! box dimension
        real(8),intent(out) :: h(n)
            !! radial distribution

        integer i, j, k
        real(8),dimension(3) :: rj, rij, iL
        real(8) dijsq

        if ( any(n*dr > Lbox*0.5d0) ) then
            error stop "Error::compute_rdf(rdf.f90):: bin width greater tham half length of the box"
        end if

        iL = 1.0/Lbox
        do j = 1, n_np
            rj(:) = r_np(:,j)
            do i = 1, nsol
                rij(:) = rj(:) - rsol(:,i)
                ! minimum image convention
                rij(:) = rij(:) - nint(rij(:)*iL(:))*Lbox(:)
                dijsq = sum(rij**2)
                k = floor(sqrt(dijsq)/dr) + 1
                if ( k <= n ) h(k) = h(k) + 1 ! add 1 because only one NP is considered
            end do            
        end do
            
    end subroutine

    subroutine compute_rdf(n, natom, nstep, rho, dr, h, gr)
        integer(4),intent(in) :: n
            !! number of histogram
        integer(4),intent(in) :: natom
            !! number of atoms
        integer(4),intent(in) :: nstep
            !! number of timestep
        real(8),intent(in) :: rho
            !! number density
        real(8),intent(in) :: dr
            !! width of bins
        real(8),intent(in) :: h(n)
            !! radial distribution histogram
        real(8),intent(out) :: gr(n)
            !! radial distribution function
    
        real(8) const, rlo, rhi, nid
        real(8) :: PI = acos(-1.d0)
        integer k

        const = 4.d0*PI*rho/3.d0
        do k = 1, n
            gr(k) = h(k)/real(natom*nstep, 8)
            rlo = real(k-1, 8)*dr
            rhi = rlo + dr
            nid = const*(rhi*rhi*rhi - rlo*rlo*rlo)
            gr(k) = gr(k)/nid
        end do

    end subroutine
end module rdf_m