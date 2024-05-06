import math

class MPCDConfiguration:
    def __init__(self, mass:float, num_density:int, kT:float, dx:float, dt:float, period:int, alpha:float) -> None:
        self.mass = mass
        self.M = num_density
        self.kT = kT
        self.dx = dx
        self.dt_md = dt
        self.period = period
        self.alpha = alpha
        
        self.dt_mpcd = dt*period
        self.density = self.M*self.mass


    @property
    def mean_free_path(self):
        return self.speed_of_sound*self.dt_mpcd
    
    @property
    def speed_of_sound(self):
        return math.sqrt(self.kT/self.mass)

    def calc_solute_particle_mass(self):
        return self.density*self.dx**3.0

    @property
    def kinematic_viscosity(self):
        _fact = self.M-1+math.exp(-self.M)
        _rad = math.radians(self.alpha)
        _kin = 5.0*self.M/((_fact)*(2.0-math.cos(_rad)-math.cos(2.0*_rad))) - 1.0
        _col = _fact/(6.0*3.0*self.M)*(1.0 - math.cos(_rad))
        return 0.5*self.kT*self.dt_mpcd/self.mass*_kin + self.dx**2.0/self.dt_mpcd*_col

    @property
    def diffusivity(self):
        _fact = self.M-1+math.exp(-self.M)
        _rad = math.radians(self.alpha)
        return 0.5*self.kT*self.dt_mpcd/self.mass*(3.0*self.M/(_fact*(1.0-math.cos(_rad)))-1.0)

    @property
    def Schmidt_number(self):
        return self.kinematic_viscosity/self.diffusivity


    def Reynolds_number(self, vel:float, length:float):
        return vel*length/self.kinematic_viscosity

    def is_compressible(self, vel:float, tolerance:float):
        """
        Validate whether the fluid is compressible or not.
        
        The fluid is incompressible if Ma = vel/C < tolerance, where C: speed of sound.

        """
        return vel > tolerance*self.speed_of_sound

if __name__ == "__main__":
    mpcd_config = MPCDConfiguration(1.0, 5, 1.0, 1.0, 0.002, 50, 130.0)
    print(mpcd_config.density)
    print(mpcd_config.kinematic_viscosity)
    print(mpcd_config.diffusivity)
    print(mpcd_config.Schmidt_number)