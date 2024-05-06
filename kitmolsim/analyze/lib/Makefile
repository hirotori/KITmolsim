# Makefile for compiling Fortran code into a shared library

# Compiler settings
FC = gfortran
FFLAGS = -fPIC -O2

# Shared library settings
LIB_NAME = libf90_distance.so

# Source files
SRC = distance.f90

# Object files
OBJ = $(SRC:.f90=.o)

all: $(LIB_NAME)

$(LIB_NAME): $(OBJ)
	$(FC) -shared -o $@ $(OBJ)

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

clean:
	rm -f $(OBJ) $(LIB_NAME)
