# inspiration
# https://stackoverflow.com/questions/35234003/how-to-create-a-makefile-for-a-fortran-program-using-modules
F2PY = f2py
FVENDOR = gnu95
FC = gfortran


all: main

main: dpd_io.o dpd_f.o main.o
	${FC} -o $@ $^

%.o: %.f90
	${FC} -c $<

f2py: dpd_f.f90
	${F2PY} -c $< --fcompiler=${FVENDOR} -m $@


clean:
	rm -rf *.mod
	rm -rf *.so
	rm -rf *.o
	rm -rf main
