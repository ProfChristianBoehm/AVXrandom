/* This header file contains a few utils for memory allocation and time stopping */

#ifndef MYUTILS_H
#define MYUTILS_H

#include <time.h>
#include <stdlib.h>

double * mallocA64(size_t s) {
    long long adr = (long long) malloc(s + 72);
    long long adr2 = (adr + 71) & ~63;
    ((long long *) adr2)[-1] = adr;
    return (double *) adr2;
}

double * callocA64(size_t s) {
    long long adr = (long long) calloc(s + 72, 1);
    long long adr2 = (adr + 71) & ~63;
    ((long long *) adr2)[-1] = adr;
    return (double *) adr2;
}

void freeA64(void * adr) {
    free((void *) (((long long *) adr)[-1]));
}

#define MALLOCA64(number,type) ((type *) mallocA64((number)*sizeof(type)))
#define CALLOCA64(number,type) ((type *) callocA64((number)*sizeof(type)))
#define FREEA64(adr) (freeA64(adr))

double stopcH(int reset) {
    static double last;
    double c = clock();
    double r = c - last;
    if(reset) 
        last = c;
    return r / CLOCKS_PER_SEC;
}
double stopc(void){
    return stopcH(1);
}
double lapc(void){
    return stopcH(0);
}

double stopH(int reset) {
    static struct timeval last;
    struct timeval c;
    gettimeofday(&c, NULL);
    double r = (double) (c.tv_sec - last.tv_sec) + (double) (c.tv_usec - last.tv_usec) / 1000000.;
    if(reset){
        last.tv_sec = c.tv_sec;
        last.tv_usec = c.tv_usec;
    }
    return r;
}
double stop(void) {
    return stopH(1);
}
double lap(void) {
    return stopH(0);
}

double stoptH(int reset) {
    static unsigned long long last, c;
    c = __rdtsc();
    double r = (double)(c - last);
    if(r<0) r+= 18446744073709551616.;
    if(reset) last = c;
    return r;
}
double stopt(){return stoptH(1);}
double lapt(){return stoptH(0);}

#endif /* MYUTILS_H */

