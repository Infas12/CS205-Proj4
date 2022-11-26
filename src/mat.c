#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

simple_mat* create_empty_matrix(int row, int column)
{
    simple_mat* p_mat = (simple_mat*) malloc (sizeof(simple_mat));

    p_mat -> row = row;
    p_mat -> column = column;
    p_mat -> data = (float*) malloc (row*column*sizeof(float));

    for(int i = 0 ;i < row*column; i++){
        p_mat -> data [i] =  0.0;
    }

    return p_mat;
}

simple_mat* create_random_matrix(int row, int column)
{
    // srand(time(0));
    simple_mat* p_mat = (simple_mat*) malloc (sizeof(simple_mat));
    
    p_mat -> row = row;
    p_mat -> column = column;
    p_mat -> data = (float*) malloc (row*column*sizeof(float));

    for(int i = 0 ;i < row*column; i++){
        p_mat -> data [i] =  rand() * (2.0 / RAND_MAX) - 1.0; 
    }

    return p_mat;
}

void delete_matrix(simple_mat** pp_mat)
{
    float** pp_data = &((*pp_mat) -> data);
    free(*pp_data);
    free(*pp_mat);
    *pp_mat = NULL;
}

void print_matrix(simple_mat* p_mat)
{
    for(int i = 0; i < p_mat->row; i++){
        printf("[");
        for(int j = 0; j < p_mat->column; j++){
            printf("%3f,",p_mat->data[i*p_mat->column+j]);
        }
        printf("]\n");
    }
}

simple_mat* matmul_plain(simple_mat* left_op, simple_mat* right_op)
{

    int result_row = left_op->row;
    int result_col = right_op->column;

    simple_mat* result = create_empty_matrix(result_row,result_col);

    for(int i = 0; i < result_row; i++){
        for(int j = 0; j < result_col; j++){
            result->data[i*result_col+j] = 0.0;
            for(int k=0;k<left_op->column;k++){    
                result->data[i*result_col+j] += 
                    left_op->data[i*left_op->column+k] * right_op->data[k*right_op->column+j];
            }    
        }    
    }

    return result;

}

simple_mat* matmul_with_transpose(simple_mat* left_op, simple_mat* right_op)
{

    simple_mat* right_op_transpose = get_transpose_matrix(right_op);

    int result_row = left_op->row;
    int result_col = right_op->column;

    int result_size = result_row * result_col;
    simple_mat* result = create_empty_matrix(result_row,result_col);    

    __m256 avx_zeros = _mm256_setzero_ps();

    #pragma omp parallel for
    for(int i = 0; i < result_row; i++){

        #pragma omp parallel for
        for(int j = 0; j < result_col; j++){

            result->data[i*result_col+j] = 0.0;
            int res = left_op -> column % 8;
            
            #pragma omp parallel for
            for(size_t k = 0; k < left_op->column - res; k += 8){
                __m256 left, right;
                __m256 result_buf_mem = _mm256_setzero_ps();
                float result_buf_arr[8] = {0};

                left  = _mm256_loadu_ps(left_op->data + i*left_op->column+k);
                right = _mm256_loadu_ps(right_op_transpose->data + j*right_op_transpose->column+k);
                result_buf_mem = _mm256_add_ps(result_buf_mem,_mm256_mul_ps(left, right));
                
                result_buf_mem = _mm256_hadd_ps(result_buf_mem,avx_zeros);
                result_buf_mem = _mm256_hadd_ps(result_buf_mem,avx_zeros);

                _mm256_storeu_ps(result_buf_arr, result_buf_mem);
                result->data[i*result_col+j] += (result_buf_arr[0] + result_buf_arr[4]);
                result_buf_mem = _mm256_setzero_ps();
            }
            
            #pragma omp parallel for
            for(size_t k = left_op->column - res; k < left_op->column; k++){
                result->data[i*result_col+j] += 
                    left_op->data[i*left_op->column+k] * right_op_transpose->data[j*right_op_transpose->column+k];
            }

        }    
    }

    delete_matrix(&right_op_transpose);

    return result;

}



//for dense matrix
simple_mat* get_transpose_matrix(simple_mat* mat)
{

    int result_row = mat->column;
    int result_column = mat->row;

    int result_size = result_row * result_column;
    simple_mat* result = create_empty_matrix(result_row,result_column);

    for(int i = 0; i < result_row; i++){
        for(int j = 0; j < result_column; j++){
            result->data[i*result_column+j] = mat->data[j*result_row+i];
        }    
    }

    return result;
}

simple_mat* matmul_BLAS(simple_mat* left_op, simple_mat* right_op)
{

    int result_row = left_op->row;
    int result_col = right_op->column;

    simple_mat* result = create_empty_matrix(result_row,result_col);

    int M = left_op->row;
    int N = right_op->column;
    int K = left_op->column;
    float alpha = 1;
    float beta = 0;
    int lda = M;
    int ldb = K;
    int ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, left_op->data, lda, right_op->data, ldb, beta, result->data, ldc);

    return result;
    
}