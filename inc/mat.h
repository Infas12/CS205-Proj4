#ifndef MAT_H
#define MAT_H

#include <stdlib.h>
#include <openblas/cblas.h>

typedef struct {

    int row;
    int column;
    float* data;

} simple_mat;

simple_mat* create_empty_matrix(int row, int column);

simple_mat* create_random_matrix(int row, int column);

simple_mat* matmul_with_transpose(simple_mat* left_op, simple_mat* right_op);

simple_mat* matmul_plain(simple_mat* left_op, simple_mat* right_op);

simple_mat* matmul_BLAS(simple_mat* left_op, simple_mat* right_op);

simple_mat* get_transpose_matrix(simple_mat* mat);

void delete_matrix(simple_mat** p_mat);

void print_matrix(simple_mat* p_mat);

#endif