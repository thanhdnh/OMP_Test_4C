#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

const long NUM_STEP = 100000;

double f(double x)
{
    return sin(x)*sin(x)/(x*x+1);
}

void integrate(double from, double to)
{
    double sum[NUM_STEP];
    long i = 0;
    double step = (to - from)/NUM_STEP;
    double result = 0;

    double start = clock();

    for(i=0; i<NUM_STEP; i++)
        sum[i] = step*f(from+i*step);
    for(i=0; i<NUM_STEP; i++)
        result += sum[i];

    double end = (clock() - start)/CLOCKS_PER_SEC;

    printf("No-Parallel Integration is %lf, and evaluated within %lf", result, end);
}

void integrateParallel(double from, double to, int cores)
{
    double sum[NUM_STEP];
    long i = 0;
    double step = (to - from)/NUM_STEP;
    double result = 0;

    double start = clock();

    #pragma omp parallel for shared(sum) num_threads(cores)
    for(i=0; i<NUM_STEP; i++)
        sum[i] = step*f(from+i*step);
    for(i=0; i<NUM_STEP; i++)
        result += sum[i];

    double end = (clock() - start)/CLOCKS_PER_SEC;

    printf("Parallel Integration with %d cores is %lf, and evaluated within %lf", cores, result, end);
}

void integrateParallel2(double from, double to){
    double sum[NUM_STEP];
    long i = 0;
    double step = (to - from)/NUM_STEP;
    double result = 0;

    double start = clock();
    #pragma omp parallel
    {
        #pragma omp task shared(sum)
        for(i=0; i<NUM_STEP/4; i++)//25
            sum[i] = step*f(from+i*step);
        #pragma omp task shared(sum)
        for(i=NUM_STEP/4; i<NUM_STEP/2; i++)//50
            sum[i] = step*f(from+i*step);
        #pragma omp task shared(sum)
        for(i=NUM_STEP/2; i<3*NUM_STEP/4; i++)//75
            sum[i] = step*f(from+i*step);
        #pragma omp task shared(sum)
        for(i=3*NUM_STEP/4; i<NUM_STEP; i++)
            sum[i] = step*f(from+i*step);
    }
    for(i=0; i<NUM_STEP; i++)
        result += sum[i];

    double end = (clock() - start)/CLOCKS_PER_SEC;

    printf("Parallel Integration version 2 with 4 cores is %lf, and evaluated within %lf", result, end);
}

void integrateParallel3(double from, double to, int cores)
{
    long i = 0;
    double step = (to - from)/NUM_STEP;
    double result = 0;

    double start = clock();

    #pragma omp parallel for reduction(+:result) num_threads(cores)
    for(i=0; i<NUM_STEP; i++)
        result += step*f(from+i*step);

    double end = (clock() - start)/CLOCKS_PER_SEC;

    printf("Parallel Integration version 3 with %d cores is %lf, and evaluated within %lf", cores, result, end);
}

int main()
{
    integrate(0, 1);
    printf("\n================\n");
    integrateParallel(0, 1, 4);
    printf("\n================\n");
    integrateParallel2(0, 1);
    printf("\n================\n");
    integrateParallel3(0, 1, 4);
    printf("\n================\n");
    printf("\n\n");
    return 0;
}
