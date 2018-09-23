CC=gcc

CFLAGS=-Wall -g -O3 -msse4 -fPIC
LDFLAGS=-g -Wall -O3 -msse4 -fPIC
LIBFLAGS=-lm -ljpeg -lpng
LIBAFLAGS=-static /usr/lib/x86_64-linux-gnu/libjpeg.so  /usr/lib/x86_64-linux-gnu/libpng12.so /usr/lib64/libz.a /lib/x86_64-linux-gnu/libm.so.6
CPYTHONFLAGS=-I/raid/users/oozdemir/anaconda3/bin/python3.6

SOURCES := $(shell find . -name '*.c' ! -name 'deepflow2_wrap.c')
OBJ := $(SOURCES:%.c=%.o) 
HEADERS := $(shell find . -name '*.h')

all: deepflow2

deepflow2: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBFLAGS)

deepflow2-static: $(OBJ)
	$(CC) -o $@ $^ $(LIBAFLAGS)

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $+ 

python: all
	swig -python $(CPYTHONFLAGS) deepflow2.i
	gcc $(CFLAGS) -c deepflow2_wrap.c $(CPYTHONFLAGS)
	gcc -shared $(LDFLAGS) deepflow2_wrap.o $(OBJ) -o _deepflow2.so $(LIBFLAGS) 


clean:
	rm -f *.o deepflow2 _deepflow2.so deepflow2.pyc deepflow2.mex???
