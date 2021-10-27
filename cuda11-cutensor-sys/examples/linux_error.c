// g++ examples/linux_error.c -o test -std=c++11 -I/usr/local/cuda-11.4/include -L/usr/local/cuda/lib64 -L/usr/local/cuda-ll.4/lib64 -lcudart -lcutensor 
#include <cuda_runtime.h>
#include <cutensor.h>


int main(int n_arg, char** arg){

    printf("rt version %li\n", cutensorGetCudartVersion());
    printf("cutensor version %li\n", cutensorGetVersion());

    cudaStream_t cuda_stream = NULL;
    int err = cudaStreamCreate(&cuda_stream);
    if( err != 0 ){
        printf("%i\n", err);
    }

    cutensorHandle_t cutensor_handle = {};

    err = cutensorInit(&cutensor_handle);
    if( err != 0 ){
        printf("Error cutensor init: %i\n", err);
    }

    void* workspace = NULL;
    int workspace_size = 18877456; 
    err = cudaMalloc(&workspace, workspace_size);
    if( err != 0 ){
        printf("Error Gpu Malloc %i\n", err);
    }


    int dim_labels[4] = {1, 2, 3, 4};
    int64_t dims[4] = {2, 2, 2, 2};

    int bytes = 4;
    //TODO
    for( int i = 0; i < 4; i++ ){
        bytes *= dims[i];
    }

    cutensorTensorDescriptor_t x1_descriptor = {}; //cutensorTensorDescriptor_t{ fields: [0i64; 72usize] };
    cutensorTensorDescriptor_t x2_descriptor = {}; //cutensorTensorDescriptor_t{ fields: [0i64; 72usize] };


    err = cutensorInitTensorDescriptor(
        &cutensor_handle,
        &x1_descriptor,
        4,
        dims,
        NULL,
        CUDA_R_32F,
        CUTENSOR_OP_IDENTITY
        //CUTENSOR_OP_SIGMOID
    );

    if( err != 0 ){
        printf("Error init tensor descriptor: %i\n", err);
    }


    err = cutensorInitTensorDescriptor(
        &cutensor_handle,
        & x2_descriptor,
        4,
        dims,
        NULL,
        CUDA_R_32F,
        CUTENSOR_OP_IDENTITY
    );
    if( err != 0 ){
        printf("Error init tensor descriptor: %i\n", err);
    }

    void* x1_data = NULL;
    void* x2_data = NULL;
    void* y_data  = NULL;
    err = cudaMalloc(&x1_data, bytes );
    err *= cudaMalloc(&x2_data, bytes);
    err *= cudaMalloc(&y_data, bytes);
    if( err != 0 ){
        printf("Error gpu malloc: %i\n", err);
    }

    float one = 1.0;
    err = cutensorElementwiseBinary(
        &cutensor_handle,

        &one,
        x1_data,
        &x1_descriptor,
        dim_labels,

        &one,
        x2_data,
        &x2_descriptor,
        dim_labels,

        y_data,
        &x2_descriptor,
        dim_labels, 

        CUTENSOR_OP_ADD,
        CUDA_R_32F,
        cuda_stream
    );
    if( err != 0 ){
        printf("Error elementwise binary: %i\n", err);
    }


}

