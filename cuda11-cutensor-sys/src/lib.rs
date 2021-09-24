mod cutensor;

pub use cutensor::*;


#[cfg(test)]
mod tests {
    use super::*;
    use cuda11_cudart_sys::{self, cudaMalloc, cudaStreamCreate, cudaMemcpyAsync, cudaStreamSynchronize, cudaFree, cudaStreamDestroy, cudaMemcpyKind};

    fn checkCudaStatus(status: cuda11_cudart_sys::cudaError_t ) {
        if status != cuda11_cudart_sys::cudaError::cudaSuccess {
            print!("cuda API failed with status \n");
            panic!();
        }
    }

    fn checkCutensorStatus(status: cutensorStatus_t ) {
        if status != cutensorStatus_t_CUTENSOR_STATUS_SUCCESS {
            print!("cuda API failed with status {:?}\n", status);
            panic!();
        }
    }

    // not a test for get driver version, but a test for linking
    #[test]
    fn link_test() {

        unsafe {
            let cutensor_version = cutensorGetVersion();
            println!("{:?}", cutensor_version);

            let cudart_version_for_cutensor = cutensorGetCudartVersion();
            println!("{:?}", cudart_version_for_cutensor);

            let mut stream: cudaStream_t = std::ptr::null_mut();
            checkCudaStatus(cudaStreamCreate(&mut stream as *mut _ as _));

            let mut handle:cutensorHandle_t = std::mem::uninitialized();
            checkCutensorStatus(cutensorInit(&mut handle as *mut _));

            let alpha: f32 = 1.0;

            let extent: Vec<i64> = vec![10, 3];
            let elems: usize = 10*3;

            let sizeA = std::mem::size_of::<f32>()*elems;
            let sizeB = std::mem::size_of::<f32>()*elems;
            
            let mut A_d: *mut f32 = std::ptr::null_mut();
            let mut B_d: *mut f32 = std::ptr::null_mut();
            checkCudaStatus(cudaMalloc(&mut A_d as *mut _ as *mut _, sizeA));
            checkCudaStatus(cudaMalloc(&mut B_d as *mut _ as *mut _, sizeB));

            let mut Ahost: Vec<f32> = vec![0. ; elems];
            let mut Bhost: Vec<f32> = vec![0. ; elems];
            for i in 0..elems {
                Ahost[i] = i as f32 + 1.0;
            }

            checkCudaStatus(cudaMemcpyAsync(A_d as *mut _, Ahost.as_ptr() as *mut _, sizeA, cudaMemcpyKind::cudaMemcpyHostToDevice, stream as _));

            let mut descA: cutensorTensorDescriptor_t = std::mem::uninitialized();
            let mut descB: cutensorTensorDescriptor_t = std::mem::uninitialized();

            checkCutensorStatus(cutensorInitTensorDescriptor( &mut handle,
                                           &mut descA,
                                           2,
                                           extent.as_ptr(),
                                           std::ptr::null(),/*stride*/
                                           cudaDataType_t::CUDA_R_32F,
                                           cutensorOperator_t_CUTENSOR_OP_IDENTITY));
            checkCutensorStatus(cutensorInitTensorDescriptor( &mut handle,
                                           &mut descB,
                                           2,
                                           extent.as_ptr(),
                                           std::ptr::null(),/*stride*/
                                           cudaDataType_t::CUDA_R_32F,
                                           cutensorOperator_t_CUTENSOR_OP_IDENTITY));

            let modeA: Vec<i32> = vec![32, 33];
            let modeB: Vec<i32> = vec![32, 33];
            
            checkCutensorStatus(cutensorPermutation(&handle,
                                &alpha as *const _ as _,
                                A_d as _,
                                &descA as _,
                                modeA.as_ptr(),
                                B_d as _,
                                &descB as _,
                                modeB.as_ptr(),
                                cudaDataType_t::CUDA_R_32F,
                                stream
            ));

            cudaStreamSynchronize(stream as _);

            checkCudaStatus(cudaMemcpyAsync(Ahost.as_mut_ptr() as *mut _, A_d as *mut _, sizeA, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream as _));
            checkCudaStatus(cudaMemcpyAsync(Bhost.as_mut_ptr() as *mut _, B_d as *mut _, sizeB, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream as _));

            cudaStreamSynchronize(stream as _);

            checkCudaStatus(cudaFree(A_d as _));
            checkCudaStatus(cudaFree(B_d as _));
            checkCudaStatus(cudaStreamDestroy(stream as _));

            println!("{:?}", Ahost);
            println!("{:?}", Bhost);
        }

    }
}
