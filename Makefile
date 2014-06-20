# Makefile for Latent Structural SVM

CC=gcc
#CFLAGS= -O3
CFLAGS= -g
LD=gcc
#LDFLAGS= -O3
LDFLAGS= -g
LIBS= -lm
#NOTE: Change the following variable to point to your Mosek headers directory
MOSEK_H= /Users/Aseem/work/mosek/6/tools/platform/osx64x86/h/
MSKLINKFLAGS= -lmosek64 -lpthread -lm
#NOTE: Change the following variable to point to your Mosek library directory
MSKLIBPATH= /Users/Aseem/work/mosek/6/tools/platform/osx64x86/bin/

all: svm_latent_learn  svm_latent_classify

clean: tidy
	rm -f svm_latent_learn svm_latent_classify

tidy:
	rm -f *.o

svm_latent_learn: svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o 
	$(LD) $(LDFLAGS) svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o -o svm_latent_learn \
	                 $(LIBS) -L $(MSKLIBPATH) $(MSKLINKFLAGS) 

svm_latent_classify: svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o 
	$(LD) $(LDFLAGS) svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o -o svm_latent_classify $(LIBS)

svm_struct_latent_spl.o: svm_struct_latent_spl.c
	$(CC) -std=c99 -c $(CFLAGS) svm_struct_latent_spl.c -o svm_struct_latent_spl.o

svm_common.o: ./svm_light/svm_common.c ./svm_light/svm_common.h ./svm_light/kernel.h
	$(CC) -c $(CFLAGS) ./svm_light/svm_common.c -o svm_common.o

mosek_qp_optimize.o: mosek_qp_optimize.c
	$(CC) -c $(CFLAGS) mosek_qp_optimize.c -o mosek_qp_optimize.o -I $(MOSEK_H)

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o

