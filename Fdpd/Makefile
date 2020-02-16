# inspiration
# https://stackoverflow.com/questions/35234003/how-to-create-a-makefile-for-a-fortran-program-using-modules
F2PY = f2py3
FVENDOR = gnu95
FC = gfortran
FLAGS = -g -fcheck=all -Wall -Wunused


all: main

run: all
	./main

main: dpd_io.o dpd_f.o mdpd_f.o main.o
	${FC} -o $@ $^

%.o: %.f90
	${FC} ${FLAGS} -c $<

f2py: dpd_f.f90 mdpd_f.f90
	${F2PY} -c dpd_f.f90 --fcompiler=${FVENDOR} -m dpd_f
	${F2PY} -c mdpd_f.f90 --fcompiler=${FVENDOR} -m mdpd_f


clean:
	rm -rf *.mod
	rm -rf *.so
	rm -rf *.o
	rm -rf main
