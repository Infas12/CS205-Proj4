#include "inc/mat.h"
#include <stdio.h>
#include <sys/time.h>


int main()
{
    struct timeval stop, start;

    

    for(int i = 100; i <= 5000; i += 100){
        
        //Improved Method
        simple_mat* left_op = create_random_matrix(i,i);    
        simple_mat* right_op = get_transpose_matrix(left_op);
        simple_mat* result;

        gettimeofday(&start, NULL);
        result = matmul_with_transpose(left_op,right_op);
        gettimeofday(&stop, NULL);
        float elapsed_optimized = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
        delete_matrix(&result);

        //Original Method
        gettimeofday(&start, NULL);
        result = matmul_plain(left_op,right_op);
        gettimeofday(&stop, NULL);
        float elapsed_raw = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
        delete_matrix(&result);
    
        //OpenBLAS
        gettimeofday(&start, NULL);
        result = matmul_BLAS(left_op,right_op);
        gettimeofday(&stop, NULL);
        float elapsed_blas = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
        delete_matrix(&result);
        

        printf("%d,%f,%f, %f\n",i,elapsed_optimized,elapsed_raw, elapsed_blas);

    }

    return 0;
}