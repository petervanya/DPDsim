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

f2py: dpd_f.f90 mdpd_f.f90 emdpd_f.f90 gmdpd_f.f90
	${F2PY} -c dpd_f.f90 --fcompiler=${FVENDOR} -m dpd_f
	${F2PY} -c mdpd_f.f90 --fcompiler=${FVENDOR} -m mdpd_f
	${F2PY} -c emdpd_f.f90 --fcompiler=${FVENDOR} -m emdpd_f
	${F2PY} -c gmdpd_f.f90 --fcompiler=${FVENDOR} -m gmdpd_f

dpdpy: dpd_f.f90
	python -m numpy.f2py -c -m dpd_f $<

mdpdpy: mdpd_f.f90
	python -m numpy.f2py -c -m mdpd_f $<

gmdpdpy: emdpd_f.f90
	python -m numpy.f2py -c -m emdpd_f $<

gmdpdpy: gmdpd_f.f90
	${F2PY} -c $< --fcompiler=${FVENDOR} -m gmdpd_f

clean:
	rm -rf *.mod
	rm -rf *.so
	rm -rf *.o
	rm -rf main
