# AVXrandom
A SIMD and MIMD parallel version of the 48-bit pseudo-random number generator drand48/erand48 in C++ and Java.util.Random using AVX-512 and OpenMP.

  File:   avxrandom.h
 
  README:
  This header file defines the AVX-512 versions of the operations drand48(), erand48(), Java.util.Random.nextDouble(),
  as well as an unbiased method to get 64-bit integers in {0, 1, ... u-1}, where u is an upper bound, along with
  the corresponding seeding methods. The generated sequences are identical to those generated by drand48, erand48,
  and nextDouble. With our seeding methods, it is also applicable in a multi-core environment (e.g. using OpenMP)
  where each core (each thread) generates different sub-sequences of the original sequence of random numbers.
  
  Method summary:
 * _mm512_drand48_pd(s): returns in an AVX-512 vector 8 doubles that are subsequently generated by drand48/erand48
 * _mm512_nextdouble_pd(s): returns in an AVX-512 vector 8 doubles that are subsequently generated by Java.util.Random.nextDouble.
 * _mm512_nextlong_epi64(u,s): returns in an AVX-512 vector 8 int64 between 0 and u-1 (the upper boundary)
 * _mm512_seed_skip_epi64(v,j): the seed method for _mm512_drand48_pd and _mm512_sextlong_epi64; returns the initial state s
 * _mm512_java_seed_skip_epi64(v,j): the seed method for _mm512_nextdouble_pd; returns the initial state s
  
  s (__m512i) is the state vector
  v (int) is the seed value as it is used in srand48 and in the constructor in java.util.Random.
  j (int64) is the number of random numbers that are implicitly skipped before the first number is generated
  
  The following code example demonstrates its use:
  srand48(25);
  for(int i=0; i<n; i++){
      double h = drand48();
      // do something with h;
  }
  
  This is parallelized using OpenMP and AVX-512 as follows:
  #pragma omp parallel for
  for(int p=0; p<omp_get_max_threads(); p++){
      __m512i s = _mm512_seed_skip_epi64(25, p*n/8/omp_get_max_threads()*8);
      for(int i=p*n/8/omp_get_max_threads(); i<(p+1)*n/8/omp_get_max_threads(); i++){
          __m512d h = _mm512_drand48_pd(s);
          // do something with h;
  }   }
  
  If you use our methods for a publication please cite the following paper:
  @InProceedings{BP2020,
    Title     = {Massively Parallel Random Number Generation},
    Author    = {Christian B\"(o)hm and Claudia Plant},
    Booktitle = {IEEE Int. Conf. on Big Data (BigData 2020)},
    Year      = {2020}}
 
