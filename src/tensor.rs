

use crate::{cuda_error, tensor_error};
use crate::model::*;
use cuda11_cudnn_sys::*;

use cuda11_cutensor_sys::{cutensorGetErrorString, cutensorInit, cutensorHandle_t, 
cutensorTensorDescriptor_t, cutensorOperator_t_CUTENSOR_OP_IDENTITY, cutensorOperator_t_CUTENSOR_OP_SIGMOID, cutensorOperator_t_CUTENSOR_OP_EXP, cutensorOperator_t_CUTENSOR_OP_ADD, cutensorOperator_t_CUTENSOR_OP_MUL, cutensorInitTensorDescriptor, cutensorElementwiseBinary};



use std::ptr::{null, null_mut};
use std::mem::size_of;
use std::mem::transmute;




pub struct GpuTensorTensor{
    pub init: bool,
    pub dims: [i32;4],
    pub descriptor : cutensorTensorDescriptor_t,
    pub data: *mut std::ffi::c_void,
}
impl Default for GpuTensorTensor{
    fn default()->Self{
        GpuTensorTensor::new() 
    }
}

impl GpuTensorTensor{
    //TODO drop
    pub fn new()->Self{
        GpuTensorTensor{
            init: false,
            dims: [0i32;4],
            descriptor : cutensorTensorDescriptor_t{ fields: [0i64; 72usize] },
            data: null_mut(),
        }
    }
    pub fn size_bytes(&self)->usize{
        let mut x = 4;
        for (i, it) in self.dims.iter().enumerate(){
            x *= *it;
        }
        x as _
    }
    pub fn construct(cutensor_handle: &cutensorHandle_t, dims: &[i32], op: GpuTensorOps)->Self{

        let mut rt = Self::new();
        rt.init = true;

        rt.initialize_desciptor(cutensor_handle, dims, op);
        
        let size = {
            let mut x = 4;
            for (i, it) in dims.iter().enumerate(){
                x *= *it;
                rt.dims[i] = *it;
            }
            x
        };
        rt.gpu_alloc(size as _);

        rt
    }
    pub fn gpu_alloc(&mut self, bytes: usize){
        cuda_error!(cudaMalloc(&mut self.data, bytes as _));
    }

    pub fn initialize_desciptor(&mut self, cutensor_handle: &cutensorHandle_t, dims: &[i32], op: GpuTensorOps){
        let mut _dims = vec![];
        for it in dims.iter(){
            _dims.push(*it as i64);
        }

        tensor_error!(cutensorInitTensorDescriptor(
            cutensor_handle,
            &mut self.descriptor,
            dims.len() as _,
            _dims.as_ptr(),
            null(),
            cuda11_cutensor_sys::cudaDataType_t::CUDA_R_32F,
            op as _,
        ));
    }
}



pub enum GpuTensorOps{
    Identity = cutensorOperator_t_CUTENSOR_OP_IDENTITY as isize,
    //pub const cutensorOperator_t_CUTENSOR_OP_SQRT: cutensorOperator_t = 2;
    //pub const cutensorOperator_t_CUTENSOR_OP_RELU: cutensorOperator_t = 8;
    //pub const cutensorOperator_t_CUTENSOR_OP_CONJ: cutensorOperator_t = 9;
    //pub const cutensorOperator_t_CUTENSOR_OP_RCP: cutensorOperator_t = 10;
    Sigmoid = cutensorOperator_t_CUTENSOR_OP_SIGMOID as isize,
    //pub const cutensorOperator_t_CUTENSOR_OP_TANH: cutensorOperator_t = 12;
    Exp = cutensorOperator_t_CUTENSOR_OP_EXP as isize,
    //pub const cutensorOperator_t_CUTENSOR_OP_LOG: cutensorOperator_t = 23;
    //pub const cutensorOperator_t_CUTENSOR_OP_ABS: cutensorOperator_t = 24;
    //pub const cutensorOperator_t_CUTENSOR_OP_NEG: cutensorOperator_t = 25;
    //pub const cutensorOperator_t_CUTENSOR_OP_SIN: cutensorOperator_t = 26;
    //pub const cutensorOperator_t_CUTENSOR_OP_COS: cutensorOperator_t = 27;
    //pub const cutensorOperator_t_CUTENSOR_OP_TAN: cutensorOperator_t = 28;
    //pub const cutensorOperator_t_CUTENSOR_OP_SINH: cutensorOperator_t = 29;
    //pub const cutensorOperator_t_CUTENSOR_OP_COSH: cutensorOperator_t = 30;
    //pub const cutensorOperator_t_CUTENSOR_OP_ASIN: cutensorOperator_t = 31;
    //pub const cutensorOperator_t_CUTENSOR_OP_ACOS: cutensorOperator_t = 32;
    //pub const cutensorOperator_t_CUTENSOR_OP_ATAN: cutensorOperator_t = 33;
    //pub const cutensorOperator_t_CUTENSOR_OP_ASINH: cutensorOperator_t = 34;
    //pub const cutensorOperator_t_CUTENSOR_OP_ACOSH: cutensorOperator_t = 35;
    //pub const cutensorOperator_t_CUTENSOR_OP_ATANH: cutensorOperator_t = 36;
    //pub const cutensorOperator_t_CUTENSOR_OP_CEIL: cutensorOperator_t = 37;
    //pub const cutensorOperator_t_CUTENSOR_OP_FLOOR: cutensorOperator_t = 38;
    Add = cutensorOperator_t_CUTENSOR_OP_ADD as isize,
    Mul = cutensorOperator_t_CUTENSOR_OP_MUL as isize,
    //pub const cutensorOperator_t_CUTENSOR_OP_MAX: cutensorOperator_t = 6;
    //pub const cutensorOperator_t_CUTENSOR_OP_MIN: cutensorOperator_t = 7;
    //pub const cutensorOperator_t_CUTENSOR_OP_UNKNOWN: cutensorOperator_t = 126;
}

//TODO document the behavior of this function. It will not be clear to new users 
pub fn calc_elementwise_binary(cutensor_handle: &cutensorHandle_t, cuda_stream: cudaStream_t, dim_labels: &[char], x1: &GpuTensorTensor, x2: &mut GpuTensorTensor, _y: Option<&mut GpuTensorTensor>, op: GpuTensorOps, alpha: f32, beta: f32){

    let y_data = match _y {
        Some(a)=>{ a.data },
        _=>{ x2.data }
    };
    tensor_error!(cutensorElementwiseBinary(
        cutensor_handle,

        (&alpha as *const f32) as _,
        x1.data,
        &x1.descriptor,
        dim_labels.as_ptr() as _,

        (&beta as *const f32) as _,
        x2.data,
        &x2.descriptor,
        dim_labels.as_ptr() as _,

        y_data,
        //&y.descriptor, //isn't this identical?
        &x2.descriptor,
        dim_labels.as_ptr() as _, 

        op as _,
        cuda11_cutensor_sys::cudaDataType_t::CUDA_R_32F,
        cuda_stream as _,
    ));

}
