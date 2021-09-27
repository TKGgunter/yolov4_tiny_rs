extern crate cuda11_cutensor_sys;

use std::ptr::{null, null_mut};
use cuda11_cutensor_sys::*;


fn main(){unsafe{
    let mut cuda_stream = null_mut();
    let err = cudaStreamCreate(&mut cuda_stream);
    if err as i32 != 0 {
        println!("{:?}", err);
    }

    let mut cutensor_handle = cutensorHandle_t {
        fields: [0i64; 512usize],
    };
    let err = cutensorInit(&mut cutensor_handle);
    if err as i32 != 0 {
        println!("Error cutensor init: {:?}", err);
    }

    let mut workspace = null_mut();
    let workspace_size = 18877456; 
    let err = cudaMalloc(&mut workspace as _, workspace_size);
    if err as i32 != 0 {
        println!("Error Gpu Malloc {:?}", err);
    }


    let dim_labels = ['a' as i32, 'b' as i32, 'c' as i32, 'd' as i32];
    let dims = [2, 2, 2, 2];

    let mut bytes = 0;
    for it in dims.iter(){
        bytes += *it
    }
    bytes *= 4;

    let mut x1_descriptor = cutensorTensorDescriptor_t{ fields: [0i64; 72usize] };
    let mut x2_descriptor = cutensorTensorDescriptor_t{ fields: [0i64; 72usize] };


    let err = cutensorInitTensorDescriptor(
        &cutensor_handle,
        &mut x1_descriptor,
        dims.len() as _,
        dims.as_ptr(),
        null(),
        cudaDataType_t::CUDA_R_32F,
        cutensorOperator_t_CUTENSOR_OP_SIGMOID,
    );
    if err as i32 != 0 {
        println!("Error init tensor descriptor: {:?}", err);
    }


    let err = cutensorInitTensorDescriptor(
        &cutensor_handle,
        &mut x2_descriptor,
        dims.len() as _,
        dims.as_ptr(),
        null(),
        cudaDataType_t::CUDA_R_32F,
        cutensorOperator_t_CUTENSOR_OP_IDENTITY,
    );
    if err as i32 != 0 {
        println!("Error init tensor descriptor: {:?}", err);
    }

    let mut x1_data = null_mut();
    let mut x2_data = null_mut();
    let mut y_data  = null_mut();
    let mut err = cudaMalloc(&mut x1_data, bytes as _) as i32;
    err *= cudaMalloc(&mut x2_data, bytes as _) as i32;
    err *= cudaMalloc(&mut  y_data, bytes as _) as i32;
    if err as i32 != 0 {
        println!("Error gpu malloc: {:?}", err);
    }

    let err = cutensorElementwiseBinary(
        &cutensor_handle,

        (&1f32 as *const f32) as _,
        x1_data,
        &x1_descriptor,
        dim_labels.as_ptr() as _,

        (&1f32 as *const f32) as _,
        x2_data,
        &x2_descriptor,
        dim_labels.as_ptr() as _,

        y_data,
        &x2_descriptor,
        dim_labels.as_ptr(), 

        cutensorOperator_t_CUTENSOR_OP_ADD,
        cuda11_cutensor_sys::cudaDataType_t::CUDA_R_32F,
        cuda_stream as _,
    );
    if err != 0 {
        println!("Error elementwise binary: {:?}", err);
    }


}}

