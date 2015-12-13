CFLAGS=-g -Wall -O0  -I/usr/local/include -I/usr/local/cuda-6.5/include
LOADLIBES=-lgsl -L/usr/lib/atlas-base/ -lopenblas -lcublas -L/opt/OpenBlas/lib -L/usr/local/cuda-6.5/lib64
test: matrix.o
clean:
	rm matrix.o test
