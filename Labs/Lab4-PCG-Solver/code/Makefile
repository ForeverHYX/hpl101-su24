CC = gcc
CFLAGS = -fopenmp -O3

.PHONY = pcg

pcg: main.o judge.o pcg.o
	$(CC) $(CFLAGS) $^ -o $@

judge.o: judge.c judge.h
	$(CC) $(CFLAGS) -c judge.c -o judge.o

main.o: main.c judge.h
	$(CC) $(CFLAGS) -c main.c -o main.o

pcg.o: pcg.c judge.h
	$(CC) $(CFLAGS) -c pcg.c -o pcg.o

clean:
	rm -f judge.o main.o pcg.o pcg