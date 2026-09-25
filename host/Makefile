LIB := ../firmware/lib/tinyai
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic

tinyai: main.c $(LIB)/tinyai.c $(LIB)/gate.c $(LIB)/calc.c $(LIB)/convert.c $(LIB)/tinyai.h
	$(CC) $(CFLAGS) -D_POSIX_C_SOURCE=200809L -I$(LIB) -o $@ main.c $(LIB)/tinyai.c $(LIB)/gate.c $(LIB)/calc.c $(LIB)/convert.c -lm

clean:
	rm -f tinyai

.PHONY: clean
