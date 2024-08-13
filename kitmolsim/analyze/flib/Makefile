# Makefile for compiling Fortran code into a shared library

# Compiler settings
FC = gfortran
FFLAGS = -fPIC -O2

# Shared library settings
LIB_NAME = libflib.so

MODDIR = mod

# Source files
SRC = distance.f90 rdf.f90 py_interface.f90

# Object files
OBJ = $(SRC:.f90=.o)

all: $(LIB_NAME)

$(LIB_NAME): $(OBJ)
	$(FC) -shared -o $@ $(OBJ)

%.o: %.f90
	$(FC) $(FFLAGS) -J ${MODDIR} -c $<

clean:
	rm -f $(OBJ) $(LIB_NAME)
	rm -f *.o
	rm -f ${MODDIR}/*.mod