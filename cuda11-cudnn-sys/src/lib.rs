mod cudnn;

pub use cudnn::*;


#[cfg(test)]
mod tests {
    use super::*;
    use cuda11_cudart_sys::{self, cudaMalloc, cudaStreamCreate, cudaMemcpy, cudaStreamSynchronize, cudaFree, cudaStreamDestroy, cudaMemcpyKind};

    fn checkCudaStatus(status: cuda11_cudart_sys::cudaError_t ) {
        if status != cuda11_cudart_sys::cudaError::cudaSuccess {
            print!("cuda API failed with status \n");
            panic!();
        }
    }

    fn checkCudnnStatus(status:  cudnnStatus_t) {
        if status != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            print!("cudnn API failed with status \n");
            panic!();
        }
    }

    // not a test for get driver version, but a test for linking
    #[test]
    fn link_test() {
        struct CudaTensor {
            device_data: *mut f32,
            dim: Vec<usize>,
            mm_data: Vec<f32>,
        }
        impl CudaTensor {
            fn new() -> CudaTensor {
                CudaTensor {
                    device_data: std::ptr::null_mut(),
                    dim: Vec::new(),
                    mm_data: Vec::new(),
                }
            }
            fn new_raw(data: &[f32], shape: &[usize]) -> CudaTensor {
                
                let mut device_data: *mut f32 = std::ptr::null_mut();
                let elems: usize = shape.iter().product();
                if elems != data.len() {
                    panic!();
                }

                unsafe {
                    println!("cudaMalloc");
                    checkCudaStatus(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                               std::mem::size_of::<f32>()*elems));
                    println!("cudaMemcpy");
                    cudaMemcpy(device_data as *mut _,
                               data.as_ptr() as *mut _,
                               std::mem::size_of::<f32>()*elems,
                               cudaMemcpyKind::cudaMemcpyHostToDevice);
                }
                
                CudaTensor {
                    device_data: device_data,
                    dim: shape.to_vec(),
                    mm_data: data.to_vec(),
                }
            }
            
            fn _sync(&mut self) {
                let elems: usize = self.dim.iter().product();
                
                unsafe {
                    cudaMemcpy(self.mm_data.as_mut_ptr() as *mut _,
                               self.device_data as *mut _,
                               std::mem::size_of::<f32>()*elems,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost);
                }
            }

        }
        impl Drop for CudaTensor {
            fn drop(&mut self) {
                if self.device_data != std::ptr::null_mut() {
	            unsafe {
                        println!("cudaFree");
                        checkCudaStatus(cudaFree(self.device_data as _));                    
                    }
                }
            }
        }
        impl std::fmt::Debug for CudaTensor {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

                write!(f, "{:?}\n", self.dim)?;
                write!(f, "{:?}", self.mm_data)

            }
        }

        struct CudaConv {
            a: f32,
        }
        impl CudaConv {
            fn new() -> CudaConv {
                CudaConv {
                    a: 0.,
                }
            }
            fn forward(&self,
                       input: &CudaTensor,
                       filter: &CudaTensor,
                       alpha: f32,
                       beta: f32,
            ) -> CudaTensor {
                unsafe {

                    // create cudnn handle
                    let mut cudnnHandle: cudnnHandle_t = std::ptr::null_mut();
                    checkCudnnStatus(cudnnCreate(&mut cudnnHandle));

                    // descriptors
                    let mut srcTensorDesc: cudnnTensorDescriptor_t = std::ptr::null_mut();
                    let mut dstTensorDesc: cudnnTensorDescriptor_t = std::ptr::null_mut();
                    let mut biasTensorDesc: cudnnTensorDescriptor_t = std::ptr::null_mut();

                    let mut filterDesc: cudnnFilterDescriptor_t = std::ptr::null_mut();

                    let mut  convDesc: cudnnConvolutionDescriptor_t = std::ptr::null_mut();

                    // init descriptors
                    checkCudnnStatus(cudnnCreateTensorDescriptor(&mut srcTensorDesc));
                    checkCudnnStatus(cudnnCreateTensorDescriptor(&mut dstTensorDesc));
                    checkCudnnStatus(cudnnCreateTensorDescriptor(&mut biasTensorDesc));

                    checkCudnnStatus(cudnnCreateFilterDescriptor(&mut filterDesc));

                    checkCudnnStatus(cudnnCreateConvolutionDescriptor(&mut convDesc));


                    // set descriptors
                    checkCudnnStatus(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                               cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
                                               cudnnDataType_t_CUDNN_DATA_FLOAT,
                                               input.dim[0] as _,
                                               input.dim[1] as _,
                                               input.dim[2] as _,
                                               input.dim[3] as _));

                    checkCudnnStatus(cudnnSetFilterNdDescriptor(filterDesc,
                                               cudnnDataType_t_CUDNN_DATA_FLOAT,
                                               cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
                                               4,
                                               vec![1, 1, 3, 3].as_ptr()));
                    
                    (cudnnSetConvolutionNdDescriptor(convDesc,
                                                    2,
                                                    vec![1, 1].as_ptr(),
                                                    vec![1, 1].as_ptr(),
                                                    vec![1, 1].as_ptr(),
                                                    cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
                                                    cudnnDataType_t_CUDNN_DATA_FLOAT));
                    
                    let mut tensorOuputDimA: Vec<i32> = vec![1,1,1,1];
                    checkCudnnStatus(cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                          srcTensorDesc,
                                                          filterDesc,
                                                          4,
                                                          tensorOuputDimA.as_mut_ptr()));
                    println!("tensorOuputDimA shape: {:?}", tensorOuputDimA);

                    let mut dst_data: *mut f32 = std::ptr::null_mut();
                    let output_elem: i32 = (&tensorOuputDimA).iter().product();
                    checkCudaStatus(cudaMalloc(&mut dst_data as *mut _ as *mut _,
                                               std::mem::size_of::<f32>()*(output_elem as usize)));
                    
                    //setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
                    //setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
                    checkCudnnStatus(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                                cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
                                                                cudnnDataType_t_CUDNN_DATA_FLOAT,
                                                                tensorOuputDimA[0],
                                                                tensorOuputDimA[1],
                                                                tensorOuputDimA[2],
                                                                tensorOuputDimA[3],));
  
                    let mut returnedAlgoCount: i32 = 0;
                    let mut one_algo: cudnnConvolutionFwdAlgoPerf_t = std::mem::uninitialized();
                    let mut results: Vec<cudnnConvolutionFwdAlgoPerf_t> = vec![one_algo; (cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_COUNT as usize) * 2];
                    checkCudnnStatus(cudnnFindConvolutionForwardAlgorithm(cudnnHandle, 
                                                         srcTensorDesc,
                                                         filterDesc,
                                                         convDesc,
                                                         dstTensorDesc,
                                                         cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_COUNT as _,
                                                         &mut returnedAlgoCount,
                                                         results.as_mut_ptr()));
                    println!("returnedAlgoCount: {:?}", returnedAlgoCount);


                    
                    let mut sizeInBytes: usize = 10;
                    checkCudnnStatus(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                             srcTensorDesc,
                                                                             filterDesc,
                                                                             convDesc,
                                                                             dstTensorDesc,
                                                                             results[0].algo,
                                                                             &mut sizeInBytes));
                    println!("sizeInbytes: {:?}", sizeInBytes);

                    
                    let mut workspace: *mut f32 = std::ptr::null_mut();
                    cudaMalloc(&mut workspace as *mut _ as _, 128);
                    //
                    cudnnConvolutionForward(cudnnHandle,
                                            &alpha as *const _ as _,
                                            srcTensorDesc,
                                            input.device_data as _,
                                            filterDesc,
                                            filter.device_data as _,
                                            convDesc,
                                            results[0].algo,
                                            workspace as _,
                                            sizeInBytes,
                                            &beta as *const _ as _,
                                            dstTensorDesc,
                                            dst_data as _);

                    cudaFree(workspace as _);
                    cudaFree(dst_data as _);

                    cudnnDestroyConvolutionDescriptor(convDesc);

                    cudnnDestroyFilterDescriptor(filterDesc);
                    
                    cudnnDestroyTensorDescriptor(srcTensorDesc);
                    cudnnDestroyTensorDescriptor(dstTensorDesc);
                    cudnnDestroyTensorDescriptor(biasTensorDesc);
                    
                    cudnnDestroy(cudnnHandle);
                }
                CudaTensor::new()
            }
        }

        unsafe {

            println!("cudnn version {:?} compiled against cudart version {:?}",
                     cudnnGetVersion(),
                     cudnnGetCudartVersion());
            
            let mut stream: cudaStream_t = std::ptr::null_mut();
            checkCudaStatus(cudaStreamCreate(&mut stream as *mut _ as _));


            
            let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                                &vec![1, 1, 3, 3]);
            let mut filter = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                                 &vec![1, 1, 3, 3]);
            let mut conv = CudaConv::new();
            let mut output = conv.forward(&input, &filter, 1., 0.);

            input._sync();
            println!("{:?}", input);


            
            cudaStreamSynchronize(stream as _);
            checkCudaStatus(cudaStreamDestroy(stream as _));
            
        }

    }
}
