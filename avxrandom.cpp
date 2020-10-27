using namespace std;
void *__gxx_personality_v0;

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "myutils.h"
#include "avxrandom.h"

double testDrand48(unsigned long long n, int threads){
    int actseed = 25;
    double r=0;
#pragma omp parallel for reduction(+:r)
    for(int par = 0; par<threads; par++){
        double rr[8] __attribute__ ((aligned (64)));
        unsigned long long nstart = n/8 * par / threads * 8;
        unsigned long long nstop = n/8 * (par+1) / threads * 8;
        __m512i ss = _mm512_seed_skip_epi64(actseed, nstart);
        __m512d result = _mm512_setzero_pd();
        for (unsigned long long i = nstart ; i<nstop ; i+=8){
            result += _mm512_drand48_pd(ss);
        }
        _mm512_store_pd(rr, result);
        r += rr[0]+rr[1]+rr[2]+rr[3]+rr[4]+rr[5]+rr[6]+rr[7];
    }
    return r;
}

double testnextWhole(unsigned long long n, int threads){
    int actseed = 25;
    unsigned long long r[8];
#pragma omp parallel for reduction(+:r[:8])
    for(int par = 0; par<threads; par++){
        unsigned long long nstart = n/8 * par / threads * 8;
        unsigned long long nstop = n/8 * (par+1) / threads * 8;
        unsigned long long s = (actseed<<16)|13070ull;
        __m512i ub = _mm512_set1_epi64(20ull);
        s = skiprnd_tz(s, nstart);
        __m512i ss = _mm512_maa_epi64(_mm512_set1_epi64(s), _mm512_loadu_si512(avxDrand48SeedMultipliers), _mm512_loadu_si512(avxDrand48SeedIncrements));
        __m512i result = _mm512_setzero_si512();
        for (unsigned long long i = nstart ; i<nstop ; i+=8){
            result += _mm512_nextlong_epi64(ss, ub);
        }
        _mm512_storeu_si512(r, result);
    }
    return (double) (r[0]+r[1]+r[2]+r[3]+r[4]+r[5]+r[6]+r[7]);
}

double testNextDoubleAVX(unsigned long long n, int threads){
    int actseed = 25;
    double r[8];
#pragma omp parallel for reduction(+:r[:8])
    for(int par = 0; par<threads; par++){
        unsigned long long nstart = n/8 * par / threads * 8;
        unsigned long long nstop = n/8 * (par+1) / threads * 8;
        unsigned long long s = (actseed<<16)|13070ull;
        s = skiprnd_tz(s, nstart*2);
        __m512i ss = _mm512_maa_epi64(_mm512_set1_epi64(s), _mm512_loadu_si512(avxDrand48S4DMultipliers), _mm512_loadu_si512(avxDrand48S4DIncrements));
        __m512d result = _mm512_setzero_pd();
        for (unsigned long long i = nstart ; i<nstop ; i+=8){
            result += _mm512_nextdouble_pd(ss);
        }
        _mm512_storeu_pd(r, result);
//        printf("%2d %5.3f | %f %f %f %f %f %f %f %f\n", par,lapt()/n,r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]);
    }
    return r[0]+r[1]+r[2]+r[3]+r[4]+r[5]+r[6]+r[7];
}

double testDrand48_avxengine(unsigned long long n, int threads){
    int actseed = 25;
    double r;
#pragma omp parallel for reduction(+:r)
    for(int par = 0; par<threads; par++){
        static const double p = 3.5527136788005009293556213378906e-15;
        unsigned long long x[8] __attribute__ ((aligned (64)));
        unsigned long long nstart = n/8 * par / threads * 8;
        unsigned long long nstop = n/8 * (par+1) / threads * 8;
        unsigned long long s = (actseed<<16)|13070ull;
        s = skiprnd_tz(s, nstart);
        __m512i ss = _mm512_maa_epi64(_mm512_set1_epi64(s), _mm512_loadu_si512(avxDrand48SeedMultipliers), _mm512_loadu_si512(avxDrand48SeedIncrements));
        for (unsigned long long i = nstart ; i<nstop ; i+=8){
            ss = _mm512_maa_epi64 (ss, _mm512_set1_epi64(25214903917ull), _mm512_set1_epi64(11ull));
            _mm512_store_si512(x, ss);
            r += p*x[0]+p*x[1]+p*x[2]+p*x[3]+p*x[4]+p*x[5]+p*x[6]+p*x[7];
        }
    }
    return r;
}

double testDrand48_avxengineMarsaglia(unsigned long long n, int threads){
    int actseed = 25;
    double r;
#pragma omp parallel for reduction(+:r)
    for(int par = 0; par<threads; par++){
        static const double p = 5.4210108624275221700372640043497e-20;
        unsigned long long x[8] __attribute__ ((aligned (64)));
        unsigned long long nstart = n/8 * par / threads * 8;
        unsigned long long nstop = n/8 * (par+1) / threads * 8;
        unsigned long long s = ((actseed+par)<<16)|13070ull;
        __m512i ss = _mm512_maa_epi64(_mm512_set1_epi64(s), _mm512_loadu_si512(avxDrand48SeedMultipliers), _mm512_loadu_si512(avxDrand48SeedIncrements));
        for (unsigned long long i = nstart ; i<nstop ; i+=8){
            ss = _mm512_xor_epi64(ss,_mm512_slli_epi64(ss,21)); 
            ss = _mm512_xor_epi64(ss,_mm512_srli_epi64(ss,35)); 
            ss = _mm512_xor_epi64(ss,_mm512_slli_epi64(ss,4)); 
            _mm512_store_si512(x, ss);
            r += p*x[0]+p*x[1]+p*x[2]+p*x[3]+p*x[4]+p*x[5]+p*x[6]+p*x[7];
        }
    }
    return r;
}

double testDrand48_sema(unsigned long long n, int threads){
    int actseed = 25;
    double r;
    srand48(actseed);
#pragma omp parallel for reduction(+:r)
    for(int par = 0; par<threads; par++){
        for(unsigned long long i=n*par/threads ; i<n*(par+1)/threads ; i++){
#pragma omp critical
            {
                r += drand48();
            }
        }
    }
    return r;
}

double testErand48(unsigned long long n, int threads){
    int actseed = 25;
    double r;
#pragma omp parallel for reduction(+:r)
    for(int par = 0; par<threads; par++){
        unsigned long long istart = n*par/threads ;
        unsigned long long s = (actseed<<16)|13070ull;
        s = skiprnd_tz(s, istart);
        unsigned short xsubi[3] = {(unsigned short)s, (unsigned short)(s>>16), (unsigned short)(s>>32)};
        for(unsigned long long i=n*par/threads ; i<n*(par+1)/threads ; i++){
            r += erand48(xsubi);
        }
    }
    return r;
}

double testDrand48_reimpl(unsigned long long n, int threads){
    int actseed = 25;
    double r;
#pragma omp parallel for reduction(+:r)
    for(int par = 0; par<threads; par++){
        unsigned long long istart = n*par/threads ;
        unsigned long long s = (actseed<<16)|13070ull;
        s = skiprnd_tz(s, istart);
        for(unsigned long long i=n*par/threads ; i<n*(par+1)/threads ; i++){
            s = (s * 25214903917ull + 11ull) & 281474976710655ull;
            r += (double) s * 3.5527136788005009293556213378906e-15;
        }
    }
    return r;
}

void printAVX(__m512d v){
    static double vv[8] __attribute__ ((aligned (64)));
    _mm512_store_pd(vv,v);
    printf("%16.14f %16.14f %16.14f %16.14f\n%16.14f %16.14f %16.14f %16.14f\n\n", 
            vv[0],vv[1],vv[2],vv[3],vv[4],vv[5],vv[6],vv[7]);
}

void printAVXi(__m512i v){
    static unsigned long long vv[8] __attribute__ ((aligned (64)));
    _mm512_store_epi64(vv,v);
    printf("%8ld %8ld %8ld %8ld %8ld %8ld %8ld %8ld\n\n", 
            vv[0],vv[1],vv[2],vv[3],vv[4],vv[5],vv[6],vv[7]);
}

void printAVXhex(__m512d v){
    static unsigned long long w[8] __attribute__ ((aligned (64)));
    _mm512_store_epi64(w, _mm512_castpd_si512(v));
    printf("%16lx %16lx %16lx %16lx\n%16lx %16lx %16lx %16lx\n\n", 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
}

int main(int argc, char** argv) {
//    printf("%ld\n", skiprnd_tz(0, 3));
//    output_powers();
    printf("omp threads: %d\n", omp_get_max_threads());
    __m512i state = _mm512_seed_skip_epi64(25,0);
    printAVX(_mm512_drand48_pd(state));
    printAVX(_mm512_drand48_pd(state));
    srand48(25);
    printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f\n", drand48());
    printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f\n", drand48());
    printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f\n", drand48());
    printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f ", drand48()); printf("%16.14f\n\n", drand48());
    state = _mm512_java_seed_skip_epi64(25,0);
    printAVX(_mm512_nextdouble_pd(state));
    printAVX(_mm512_nextdouble_pd(state));
    state = _mm512_java_seed_skip_epi64(25,0);
    printAVXhex(_mm512_nextdouble_pd(state));
    printAVXhex(_mm512_nextdouble_pd(state));
    state = _mm512_seed_skip_epi64(25,0);
    printAVXi(_mm512_nextlong_epi64(_mm512_set1_epi64(7), state));
    printAVXi(_mm512_nextlong_epi64(_mm512_set1_epi64(7), state));


    stopc(), stop(), stopt();
//    for(int i=4 ; i<=64 ; i+=4){
//        omp_set_num_threads(i);
//        testmkl(800000000,i,VSL_BRNG_MCG31);
//        //      8000000000
//        printf("testMKL(MCG | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*100, stop()*100, stopt()/8e8);
//    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testnextWhole(80000000000,i);
        printf("testnextWho | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc(), stop(), stopt()/8e10);
    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testDrand48(80000000000,i);
        printf("testDrand48 | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc(), stop(), stopt()/8e10);
    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testNextDoubleAVX(80000000000,i);
        printf("NextDoubAVX | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc(), stop(), stopt()/8e10);
    }
//    for(int i=4 ; i<=64 ; i+=4){
//        omp_set_num_threads(i);
//        random_init(8000000000,i,VSL_BRNG_MCG31);
//        printf("MCG31       | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*10, stop()*10, stopt()/8e9);
//    }
//    for(int i=4 ; i<=64 ; i+=4){
//        omp_set_num_threads(i);
//        random_init(8000000000,i,VSL_BRNG_MCG59);
//        printf("MCG59       | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*10, stop()*10, stopt()/8e9);
//    }
//    for(int i=4 ; i<=64 ; i+=4){
//        omp_set_num_threads(i);
//        random_init(8000000000,i,VSL_BRNG_SFMT19937);
//        printf("SFMT19937   | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*10, stop()*10, stopt()/8e9);
//    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testDrand48_avxengine(80000000000,i);
        printf("avxengine   | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc(), stop(), stopt()/8e10);
    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testDrand48_avxengineMarsaglia(80000000000,i);
        printf("avxengMarsa | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc(), stop(), stopt()/8e10);
    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testErand48(800000000,i);
        printf("testErand48 | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*100., stop()*100., stopt()/8e8);
    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testDrand48_reimpl(8000000000,i);
        printf("testDr reim | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*10., stop()*10., stopt()/8e9);
    }
    for(int i=4 ; i<=64 ; i+=4){
        omp_set_num_threads(i);
        testDrand48_sema(8000000,i);
        printf("testD sema  | %4d | %12.0f | %12.3f | %12.5f\n", omp_get_max_threads(), stopc()*10000., stop()*10000., stopt()/8e6);
    }
    srand48(25);
    for(int i=0 ; i<5 ; i++){
        for(int j=0 ; j<8 ; j++)
            printf("%f ", drand48());
        printf("\n");
    }
}