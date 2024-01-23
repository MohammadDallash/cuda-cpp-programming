#include <stdio.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <sys/time.h>

__global__ void recurse(int level)
{
    if (level<0)
    {
        int x = 1000;
        for(int i = 0; i<1e6; i++)
        {
            if (i%2)x*=10;
            else x/=10;
        }
        return;
    }
    level--;
    if (threadIdx.x == 0) recurse<<<1,2>>>(level);
    else recurse<<<1,2>>>(level);
    
    cudaDeviceSynchronize(); //stop here in the current thread until the the above kernal is done

    __syncthreads(); //block level synchronization barrier from the parent kernal of this kernal
    return;
}

void serial_recurse(int level)
{
    if (level<0)
    {
        int x = 1000;
        for(int i = 0; i<1e6; i++)
        {
            if (i%2)x*=10;
            else x/=10;
        }
        return;
    }
    level--;
    serial_recurse(level);
    serial_recurse(level);
    return;

}


int main()
{
    struct timeval startwtime,endwtime;
    double seq_time;

    printf("Startin here\n");

    int level_1 = 10;
    gettimeofday(&startwtime,NULL);

    serial_recurse(level_1);

    gettimeofday(&endwtime,NULL);

    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
    printf("serial time = %f \n", seq_time);

    int level=10;
    
    gettimeofday(&startwtime,NULL);

    recurse<<<1,2>>>(level);
    cudaDeviceSynchronize();

    gettimeofday(&endwtime,NULL);

    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
    printf("parralel time = %f \n", seq_time);

    return 0;
}