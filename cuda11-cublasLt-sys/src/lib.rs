mod cublasLt;

pub use cublasLt::*;


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

    fn checkCublasStatus(status: cublasStatus_t) {
        if status != 0 {
            print!("cuBLAS API failed with status {:?}\n", status as i32);
            panic!();
        }
    }

    // not a test for get driver version, but a test for linking
    #[test]
    fn link_test() {
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 2;
        let N: usize = 1;

        unsafe {
            let mut Adev: *mut f32 = std::ptr::null_mut();
            let mut Bdev: *mut f32 = std::ptr::null_mut();
            let mut Cdev: *mut f32 = std::ptr::null_mut();
            let mut biasDev: *mut f32 = std::ptr::null_mut();
            
            let mut workspace: *mut std::ffi::c_void = std::ptr::null_mut();
            let workspace_size: usize = 1024 * 1024 * 4;

            let mut stream: cudaStream_t = std::ptr::null_mut();

            let mut lthandle: *mut cublasLtContext = std::ptr::null_mut();


            let mut Ahost: Vec<f32> = vec![0. ; m*k*N];
            let mut Bhost: Vec<f32> = vec![0. ; k*n*N];
            let mut Chost: Vec<f32> = vec![0. ; m*n*N];
            let mut biasHost: Vec<f32> = vec![0. ; m*N];

            {
                checkCublasStatus(cublasLtCreate(&mut lthandle as *mut _));
                
                checkCudaStatus(cudaMalloc(&mut Adev as *mut _ as *mut _, m * k * N * std::mem::size_of::<f32>()));
                checkCudaStatus(cudaMalloc(&mut Bdev as *mut _ as *mut _, n * k * N * std::mem::size_of::<f32>()));
                checkCudaStatus(cudaMalloc(&mut Cdev as *mut _ as *mut _, m * n * N * std::mem::size_of::<f32>()));
                checkCudaStatus(cudaMalloc(&mut biasDev as *mut _ as *mut _, m * N * std::mem::size_of::<f32>()));
                checkCudaStatus(cudaMalloc(&mut workspace as *mut _ as *mut _, 1024 * 1024 * 4));
                
                checkCudaStatus(cudaStreamCreate(&mut stream as *mut _ as _));
            }

            {
                for i in 0..m * k * N {
                    Ahost[i] = i as f32;
                }
                for i in 0..k * n * N {
                    Bhost[i] = i as f32;
                }
                for i in 0..m * N {
                    biasHost[i] = (i + 1) as f32;
                }

                checkCudaStatus(cudaMemcpyAsync(Adev as *mut _, Ahost.as_ptr() as *mut _, Ahost.len() * std::mem::size_of::<f32>(), cudaMemcpyKind::cudaMemcpyHostToDevice, stream as _));
                checkCudaStatus(cudaMemcpyAsync(Bdev as *mut _, Bhost.as_ptr() as *mut _, Bhost.len() * std::mem::size_of::<f32>(), cudaMemcpyKind::cudaMemcpyHostToDevice, stream as _));
                checkCudaStatus(cudaMemcpyAsync(biasDev as *mut _, biasHost.as_ptr() as *mut _, biasHost.len() * std::mem::size_of::<f32>(), cudaMemcpyKind::cudaMemcpyHostToDevice, stream as _));
                
            }

            {
                let mut operationDesc: cublasLtMatmulDesc_t = std::ptr::null_mut();
                let mut Adesc: cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut Bdesc: cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut Cdesc: cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();

                let lda = m;
                let ldb = n;
                let ldc = m;

                let mut returnedResults: i32 = 0;
                //let mut heuristicResult: *mut cublasLtMatmulHeuristicResult_t = std::ptr::null_mut();
                //let mut heuristicResult: Vec<cublasLtMatmulHeuristicResult_t> = vec![std::mem::MaybeUninit::uninit(); 1];
                //let mut heuristicResult: cublasLtMatmulHeuristicResult_t = cublasLtMatmulHeuristicResult_t {
                //    algo: cublasLtMatmulAlgo_t {
                //        data: [0; 8usize],
                //    },
                //    workspaceSize: 0,
                //    state: 0,
                //    wavesCount: 0.,
                //    reserved: [0; 4],
                //};
                let mut heuristicResult: cublasLtMatmulHeuristicResult_t = std::mem::uninitialized();
                
                checkCublasStatus(cublasLtMatmulDescCreate(&mut operationDesc as *mut _ as *mut _, cublasComputeType_t_CUBLAS_COMPUTE_32F, cudaDataType_t::CUDA_R_32F));
                checkCublasStatus(cublasLtMatmulDescSetAttribute(&mut operationDesc as *mut _ as *mut _, cublasLtMatmulDescAttributes_t_CUBLASLT_MATMUL_DESC_TRANSA, &cublasOperation_t_CUBLAS_OP_N as *const _ as *const _, std::mem::size_of::<cublasOperation_t>()));
                checkCublasStatus(cublasLtMatmulDescSetAttribute(&mut operationDesc as *mut _ as *mut _, cublasLtMatmulDescAttributes_t_CUBLASLT_MATMUL_DESC_TRANSB, &cublasOperation_t_CUBLAS_OP_N as *const _ as *const _, std::mem::size_of::<cublasOperation_t>()));

                checkCublasStatus(cublasLtMatrixLayoutCreate(&mut Adesc as *mut _, cudaDataType_t::CUDA_R_32F, m as _, k as _, lda as _));
                checkCublasStatus(cublasLtMatrixLayoutCreate(&mut Bdesc as *mut _, cudaDataType_t::CUDA_R_32F, k as _, n as _, ldb as _));
                checkCublasStatus(cublasLtMatrixLayoutCreate(&mut Cdesc as *mut _, cudaDataType_t::CUDA_R_32F, m as _, n as _, ldc as _));

                checkCublasStatus(cublasLtMatmulPreferenceCreate(&mut preference as *mut _));
                checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, cublasLtMatmulPreferenceAttributes_t_CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size as *const _ as *const _, std::mem::size_of::<usize>()));
                println!("{:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", lthandle, operationDesc, Adesc, Bdesc, Cdesc, preference, heuristicResult, &mut returnedResults as *mut _);
                checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(lthandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &mut heuristicResult as *mut _, &mut returnedResults as *mut _));
                println!("{:?}", returnedResults);

                let alpha: f32 = 2.0;
                let beta: f32 = 0.0;
                
                checkCublasStatus(cublasLtMatmul(lthandle,
                                     operationDesc,
                                     &alpha as *const _ as *const _,
                                     Adev as *const _,
                                     Adesc,
                                     Bdev as *const _,
                                     Bdesc,
                                     &beta as *const _ as *const _,
                                     Cdev as *const _,
                                     Cdesc,
                                     Cdev as *mut _,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspace_size,
                                     stream));


                if !preference.as_ref().is_none() {checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));}
                if !Cdesc.as_ref().is_none() {checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));}
                if !Bdesc.as_ref().is_none() {checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));}
                if !Adesc.as_ref().is_none() {checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));}
                if !operationDesc.as_ref().is_none() {checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));}
            }

            {
                checkCudaStatus(cudaMemcpyAsync(Chost.as_ptr() as *mut _, Cdev as *mut _, Chost.len() * std::mem::size_of::<f32>(), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream as _));
                println!("chost: {:?}", Chost);
            }

            cudaStreamSynchronize(stream as _);

            {
                checkCublasStatus(cublasLtDestroy(lthandle));
                checkCudaStatus(cudaFree(Adev as _));
                checkCudaStatus(cudaFree(Bdev as _));
                checkCudaStatus(cudaFree(Cdev as _));
                checkCudaStatus(cudaFree(biasDev as _));
                checkCudaStatus(cudaFree(workspace as _));
                checkCudaStatus(cudaStreamDestroy(stream as _));
            }
        }

    }
}
