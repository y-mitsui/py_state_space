CFLAGS=-O2  -I/usr/local/include
LOADLIBES=-lgsl -lopenblas -L/opt/OpenBlas/lib
test: matrix.o
clean:
	rm matrix.o test
