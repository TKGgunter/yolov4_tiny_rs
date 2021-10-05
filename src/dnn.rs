
use crate::{cuda_error, dnn_error};
use crate::model::*;

use cuda11_cudnn_sys;
use cuda11_cudnn_sys::*;
use std::ptr::null_mut;
use std::mem::transmute;


static mut ALLOC_COUNTER : usize = 0;
pub fn print_alloc(){unsafe{
    println!("mb alloc: {}", ALLOC_COUNTER / 1_000_000);
}}

#[allow(non_snake_case)]
pub unsafe fn cudaMalloc(data: *mut *mut std::ffi::c_void, bytes: usize)->cudaError{
    ALLOC_COUNTER += bytes;
    cuda11_cudnn_sys::cudaMalloc(data, bytes as _)
}


pub struct GpuTensorConfig{
    strides_channels: i32,
    strides_width   : i32,
    strides_height  : i32,
}
pub struct GpuTensor{
  //Follow how GpuMatrix works
    pub init : bool,
    pub data  : *mut std::ffi::c_void,
    pub descriptor: cudnnTensorDescriptor_t,
    pub dims: [usize; 3],
}

impl GpuTensor{
    pub fn new()->Self{
        GpuTensor{
            init : false,
            data  : null_mut(),
            descriptor: null_mut(),
            dims: [0usize; 3],
        }
    }

    pub fn drop(&mut self){
        if self.init == false{
            println!("This struct was not init in the standard fashion. 
                      This may result in an error during the drop.");
        }
        cuda_error!(cudaFree(self.data));
        dnn_error!(cudnnDestroyTensorDescriptor(self.descriptor));
    }

    pub fn construct(width: usize, height: usize, channels: usize)->Self{
        let mut rt = GpuTensor::new();
        rt.initialize(width, height, channels);
        rt
    }

    fn size(&self)->usize{
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    pub fn size_bytes(&self)->usize{
        4 * self.dims[0] * self.dims[1] * self.dims[2]
    }
//TODO include NdTensors

    fn set_descriptor(&mut self, width: usize, height: usize, channels: usize){
        self.dims[0] = width;
        self.dims[1] = height;
        self.dims[2] = channels;

        self.descriptor = {
            let mut descriptor : cudnnTensorDescriptor_t = null_mut();
            dnn_error!(cudnnCreateTensorDescriptor(&mut descriptor as _));
            dnn_error!(cudnnSetTensor4dDescriptor(descriptor,
                                           //cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
                                           cudnnTensorFormat_t_CUDNN_TENSOR_NHWC,
                                           cudnnDataType_t_CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                                           channels as _,
                                           height as _,
                                           width  as _));
            descriptor
        };

    }

    pub fn set_descriptorex(&mut self, width: usize, height: usize, channels: usize, params: GpuTensorConfig){
        self.descriptor = {
            let mut descriptor : cudnnTensorDescriptor_t = null_mut();

            let GpuTensorConfig{
                strides_channels,
                strides_width,
                strides_height,
            } = params;

            dnn_error!(cudnnCreateTensorDescriptor(&mut descriptor as _));
            dnn_error!(cudnnSetTensor4dDescriptorEx(
                descriptor, 
                cudnnDataType_t_CUDNN_DATA_FLOAT, 
                1, //n: ::libc::c_int,
                channels as _, 
                height as _,
                width as _,
                strides_height*height as i32, //nStride: ::libc::c_int,
                strides_channels, 
                strides_height, 
                strides_width, 
            ));
            descriptor 
        };
    }

    pub fn gpu_alloc(&mut self, bytes: usize){
        cuda_error!(cudaMalloc(&mut self.data, bytes as _));
    }

    pub fn initialize(&mut self, width: usize, height: usize, channels: usize){
        self.init = true;

        self.set_descriptor(width, height, channels);
        self.gpu_alloc((4 * channels * height * width) as _);

    }

    pub fn set_data(&mut self, input: &[f32]){

        let size = input.len() * 4;
        let _input = unsafe{ transmute::<_, &[u8]>(input) };
        cuda_error!(cudaMemcpy(self.data, _input.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    pub fn fill_with_scalar(&mut self, handle: cudnnHandle_t, input: f32){
        dnn_error!(cudnnSetTensor(
            handle,
            self.descriptor,
            self.data,
            (&input as *const f32) as _,
        ));
    }

    pub fn set_data_bytes(&mut self, input: &[u8]){

        let size = input.len();
        cuda_error!(cudaMemcpy(self.data, input.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    pub fn return_data(&self)->Vec<f32>{
        let size = self.size();
        let mut output = vec![0f32; size];
        self.get_data(&mut output);

        return output;
    }

    pub fn get_data(&self, output: &mut [f32]){
        let size = output.len() * 4;
        let mut _output = unsafe{ transmute::<_, &mut [u8]>(output) };
        cuda_error!(cudaMemcpy(_output.as_mut_ptr() as _, self.data, size, 
                               cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }

    fn get_data_bytes(&self, output: &mut [u8]){
        let size = output.len();
        cuda_error!(cudaMemcpy(output.as_mut_ptr() as _, self.data, size, 
                               cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
}

pub struct GpuActivationLayer{
    pub descriptor: cudnnActivationDescriptor_t,
    pub alt_descriptor: cudnnActivationDescriptor_t,
    pub activation: GpuDnnActivation,
    pub alpha: f32,
}


impl GpuActivationLayer {
    pub fn drop(&mut self){
        dnn_error!(cudnnDestroyActivationDescriptor(self.descriptor));
        if self.alt_descriptor != null_mut(){
            dnn_error!(cudnnDestroyActivationDescriptor(self.alt_descriptor));
        }
    }

    pub fn construct(activation: GpuDnnActivation)->Self{
        let mut a = GpuActivationLayer::new();
        a.initialize(activation);
        a.alpha = -0.3f32;
        return a;
    }
    fn new()->Self{
        GpuActivationLayer{
            descriptor: null_mut(),
            alt_descriptor: null_mut(),
            activation: GpuDnnActivation::Default,
            alpha: -0.3,
        }
    }

    fn initialize(&mut self, activation: GpuDnnActivation){
        dnn_error!(cudnnCreateActivationDescriptor(&mut self.descriptor));

        self.activation = activation;
        match activation {
            GpuDnnActivation::Softmax => {
                panic!("tensor conv softmax activation has not been handled yet.");
            },
            GpuDnnActivation::LeakyRelu => {

                dnn_error!(cudnnCreateActivationDescriptor(
                    &mut self.alt_descriptor,
                ));
                dnn_error!(cudnnSetActivationDescriptor(
                    self.alt_descriptor,
                    GpuDnnActivation::Relu as _,
                    cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
                    5f64,//TODO there is a good chance this will change when working with relu
                ));

                dnn_error!(cudnnSetActivationDescriptor(self.descriptor,
                    GpuDnnActivation::Default as _,
                    cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
                    5f64,//TODO there is a good chance this will change when working with relu
                ));
            },
            _=>{ 

                dnn_error!(cudnnSetActivationDescriptor(self.descriptor,
                    activation as _,
                    cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
                    5f64,//TODO there is a good chance this will change when working with relu
                ));
            }
        }
    }

    fn calc(&self, dnn_handle: cudnnHandle_t, workspace: *mut std::ffi::c_void, workspace_size: usize, 
            output: &mut GpuTensor){

        match self.activation {
            GpuDnnActivation::Softmax => {

                dnn_error!(cudnnSoftmaxForward(
                    dnn_handle,
                    cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_FAST,
                    cudnnSoftmaxMode_t_CUDNN_SOFTMAX_MODE_CHANNEL,
                    (&1f32 as *const _) as _, //alpha: *const ::libc::c_void,
                    output.descriptor, //xDesc: cudnnTensorDescriptor_t,
                    output.data, //x: *const ::libc::c_void,
                    (&0f32 as *const _) as _, //beta: *const ::libc::c_void,
                    output.descriptor, //yDesc: cudnnTensorDescriptor_t,
                    output.data  //y: *mut ::libc::c_void,
                ));

            },
            GpuDnnActivation::LeakyRelu => {
                //NOTE checking workspace size.
                //TODO if workspace is to small maybe allow a realloc? 
                if workspace_size < (4 * output.size()){
                    panic!("workspace is insufficient {} < {}", workspace_size, output.size()*4); 
                }

                dnn_error!(cudnnAddTensor(
                    dnn_handle, //: cudnnHandle_t,
                    (&self.alpha as *const _) as _, 
                    output.descriptor, //aDesc: cudnnTensorDescriptor_t,
                    output.data, //A: *const ::libc::c_void,
                    (&0f32 as *const _) as _, //
                    output.descriptor, //cDesc: cudnnTensorDescriptor_t,
                    workspace, //C: *mut ::libc::c_void,
                )); 
               
                
                dnn_error!(cudnnActivationForward(
                    dnn_handle,
                    self.alt_descriptor, 
                    (&1f32 as *const _) as _, //TODO this was -0.3 but that seemed wrong, need to keep watch
                    output.descriptor,//xDesc: cudnnTensorDescriptor_t,
                    workspace, //x: *const ::libc::c_void,
                    (&0f32 as *const _) as _, 
                    output.descriptor, //yDesc: cudnnTensorDescriptor_t,
                    workspace, //We are using work space to handle temp data. We need to make sure it is large enough
                ));
                
                
                dnn_error!(cudnnActivationForward(
                    dnn_handle,
                    self.alt_descriptor, 
                    (&1f32 as *const _) as _, //NOTE This factor is taken from tensorflow defaults
                    //https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU
                    output.descriptor,//xDesc: cudnnTensorDescriptor_t,
                    output.data, //x: *const ::libc::c_void,
                    (&0f32 as *const _) as _, 
                    output.descriptor, //yDesc: cudnnTensorDescriptor_t,
                    output.data, //We are using work space to handle temp data. We need to make sure it is large enough
                ));
                

                dnn_error!(cudnnAddTensor(
                    dnn_handle, //: cudnnHandle_t,
                    (&-1f32 as *const _) as _, 
                    output.descriptor, //aDesc: cudnnTensorDescriptor_t,
                    workspace, //A: *const ::libc::c_void,
                    (&1f32 as *const _) as _, //
                    output.descriptor, //cDesc: cudnnTensorDescriptor_t,
                    output.data, //C: *mut ::libc::c_void,
                )); 

            },
            _=>{ 

                dnn_error!(cudnnActivationForward(
                    dnn_handle,
                    self.descriptor, 
                    (&1f32 as *const _) as _, 
                    output.descriptor,//xDesc: cudnnTensorDescriptor_t,
                    output.data, //x: *const ::libc::c_void,
                    (&0f32 as *const _) as _, 
                    output.descriptor, //yDesc: cudnnTensorDescriptor_t,
                    output.data, //We are using work space to handle temp data. We need to make sure it is large enough
                ));
            }
        }
    }
}



pub struct ConvParams{
    pub pad_height       : i32,
    pub pad_width        : i32,
    pub vertical_stride  : i32,
    pub horizontal_stride: i32,
    pub dilation_height  : i32,
    pub dilation_width   : i32,
    //cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
    //computeType=cudnnDataType_t_CUDNN_DATA_FLOAT));
}
impl Default for ConvParams{
    fn default()->Self{
        ConvParams{
            pad_height       : 0i32,
            pad_width        : 0i32,
            vertical_stride  : 1i32,
            horizontal_stride: 1i32,
            dilation_height  : 1i32,
            dilation_width   : 1i32,
            //cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
            //computeType=cudnnDataType_t_CUDNN_DATA_FLOAT));
        }
    }
}

pub struct GpuConv{
    pub init: bool,
    pub kernel_descriptor: cudnnFilterDescriptor_t,
    pub kernel: *mut std::ffi::c_void,

    pub convolution_descriptor: cudnnConvolutionDescriptor_t,
    pub algo: cudnnConvolutionFwdAlgo_t,

    pub bias: GpuTensor,

    pub activation_descriptor: cudnnActivationDescriptor_t,
    pub activation: GpuDnnActivation,

    pub alt_activation_descriptor: cudnnActivationDescriptor_t,

    pub alpha: f32,
    pub beta: f32,
}


impl GpuConv{
    pub fn drop(&mut self){
        
        if self.init != true {
            println!("A convNN was not init in a standard fashion. 
                      Memory should be dealloc manulally. ");
            return;
        }
        dnn_error!(cudnnDestroyFilterDescriptor(self.kernel_descriptor));
        cuda_error!(cudaFree(self.kernel));

        dnn_error!(cudnnDestroyConvolutionDescriptor(
            self.convolution_descriptor
        ));
        dnn_error!(cudnnDestroyActivationDescriptor(
            self.activation_descriptor,
        ));

        if self.alt_activation_descriptor != null_mut(){
            dnn_error!(cudnnDestroyActivationDescriptor(
                self.alt_activation_descriptor,
            ));
        }
        self.bias.drop();
    }
    pub fn construct(in_channels: usize, out_channels: usize, 
                  kernel_width: usize, kernel_height: usize,  
                  activation: GpuDnnActivation, input_tensor_descriptor: cudnnTensorDescriptor_t, 
                  convolution_params: ConvParams)->Self{

        let mut c = GpuConv::new();
        c.initialize(in_channels, out_channels, 
                  kernel_width, kernel_height,  
                  activation, input_tensor_descriptor, 
                  convolution_params);

        return c;
    }
    pub fn new()->Self{
        GpuConv {
            init: true,
            kernel_descriptor: null_mut(),
            kernel           : null_mut(),

            convolution_descriptor: null_mut(),

            algo: cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, //0,

            bias: GpuTensor::new(),

            activation_descriptor: null_mut(),
            activation: GpuDnnActivation::Default,

            alt_activation_descriptor: null_mut(),

            alpha: 1f32,
            beta : 0f32,
        }
        
    }

    //TODO think about these usize types. No other struct/function uses them this is not nice.
    pub fn initialize(&mut self, in_channels: usize, out_channels: usize, 
                  kernel_width: usize, kernel_height: usize,  
                  mut activation: GpuDnnActivation, input_tensor_descriptor: cudnnTensorDescriptor_t, 
                  convolution_params: ConvParams){

        self.init = true;
        let cvp = convolution_params;
        self.kernel_descriptor = {
            let mut descriptor : cudnnFilterDescriptor_t = null_mut();
            dnn_error!(cudnnCreateFilterDescriptor(&mut descriptor as _));
            dnn_error!(cudnnSetFilter4dDescriptor(descriptor,
                                                  cudnnDataType_t_CUDNN_DATA_FLOAT,
                                                //cudnnTensorFormat_t_CUDNN_TENSOR_NHWC,
                                          /*format=*/cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
                                                  out_channels as _,
                                                  in_channels  as _,
                                                  kernel_height as _,
                                                  kernel_width  as _));
            descriptor
        };

        cuda_error!(cudaMalloc(&mut self.kernel, out_channels*in_channels*kernel_height*kernel_width*4));

        self.convolution_descriptor = {
            let mut descriptor = null_mut();
            dnn_error!(cudnnCreateConvolutionDescriptor(&mut descriptor as _));
            dnn_error!(cudnnSetConvolution2dDescriptor(descriptor,
                                                       cvp.pad_height,
                                                       cvp.pad_width,
                                                       cvp.vertical_stride,
                                                       cvp.horizontal_stride,
                                                       cvp.dilation_height,
                                                       cvp.dilation_width,
                                                       //cudnnConvolutionMode_t_CUDNN_CONVOLUTION,
                                                       cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
                                                       cudnnDataType_t_CUDNN_DATA_FLOAT));
            descriptor
        };

        {
            let mut batches = 0;
            let mut channels = 0;
            let mut h = 0;
            let mut w = 0;
            dnn_error!(cudnnGetConvolution2dForwardOutputDim(
                self.convolution_descriptor,
                input_tensor_descriptor,
                self.kernel_descriptor,
                &mut batches,
                &mut channels,
                &mut h,
                &mut w,
            ));
            self.bias.initialize(1 as _, 1 as _, out_channels as _);
        }

        
        self.activation = activation;
        match self.activation {
            GpuDnnActivation::Softmax => {
                //TODO activation = GpuDnnActivation::Default;
                panic!("tensor conv softmax activation has not been handled yet.");
            },
            GpuDnnActivation::LeakyRelu => {
                activation = GpuDnnActivation::Default;

                dnn_error!(cudnnCreateActivationDescriptor(
                    &mut self.alt_activation_descriptor,
                ));
                dnn_error!(cudnnSetActivationDescriptor(
                    self.alt_activation_descriptor,
                    GpuDnnActivation::Relu as _,
                    cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
                    5f64,//TODO there is a good chance this will change when working with relu
                ));
            },
            _=>{ }
        }

        dnn_error!(cudnnCreateActivationDescriptor(
            &mut self.activation_descriptor,
        ));
        dnn_error!(cudnnSetActivationDescriptor(
            self.activation_descriptor,
            activation as _,
            cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
            5f64,//TODO there is a good chance this will change when working with relu
        ));

        //TODO according to the docs only identity and relu work with convs currently
        //I need to test what works and what does not. Our test uses relu so I guess that is cool.
        if activation == GpuDnnActivation::Default {
            self.algo = cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        }

        /*TODO for suggested algo 
        pub fn cudnnGetConvolutionForwardAlgorithm_v7(
            handle: cudnnHandle_t,
            srcDesc: cudnnTensorDescriptor_t,
            filterDesc: cudnnFilterDescriptor_t,
            convDesc: cudnnConvolutionDescriptor_t,
            destDesc: cudnnTensorDescriptor_t,
            requestedAlgoCount: ::libc::c_int,
            returnedAlgoCount: *mut ::libc::c_int,
            perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
        ) -> cudnnStatus_t;
        */
    }


    pub fn set_bias(&mut self, data: &[u8]){
        if self.bias.data == null_mut(){
            println!("Bias data has not been allocated.");
        }
        let size = data.len();
        cuda_error!(cudaMemcpy(self.bias.data, data.as_ptr() as _, size, 
                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    pub fn set_kernel(&mut self, data: &[u8]){
        if self.kernel_descriptor == null_mut(){
            println!("Kernel descriptor has not been allocated.");
        }

        let size = data.len();

        let mut k : i32 = 0;
        let mut h : i32 = 0;
        let mut w : i32 = 0;
        let mut c : i32 = 0;
        let mut data_type = cudnnDataType_t_CUDNN_DATA_FLOAT;
        let mut tensor_organization = cudnnTensorFormat_t_CUDNN_TENSOR_NHWC; 

        dnn_error!(cudnnGetFilter4dDescriptor(
            self.kernel_descriptor, 
            &mut data_type,
            &mut tensor_organization, 
            &mut k, 
            &mut c,
            &mut h,
            &mut w,
        ));

        let _size = 4*h*w*k*c;
        if size as i32 != _size{
            panic!("Could not set kernel dimensions did not match expected {} != ({:?} x 4, {})", data.len(), (h, w, k, c), _size);
        }

        cuda_error!(cudaMemcpy(self.kernel, data.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }


    pub fn calc(&self, dnn_handle: cudnnHandle_t, workspace: *mut std::ffi::c_void, workspace_size: usize, input: &GpuTensor, output: &mut GpuTensor){

        if self.init != true{
            panic!("GpuConv has not been initialized.");
        }

        //NOTE Workspace check probably should not be done here
        let mut ws = 0usize;
        dnn_error!(cudnnGetConvolutionForwardWorkspaceSize(dnn_handle,
                                                   input.descriptor,
                                                   self.kernel_descriptor,
                                                   self.convolution_descriptor,
                                                   output.descriptor,
                                                   self.algo,
                                                   &mut ws as _));


        if ws > workspace_size{
            println!("required work space size: {:?}", ws);
            panic!("workspace is too small");
        }

        dnn_error!(cudnnConvolutionBiasActivationForward(
            dnn_handle,
            (&self.alpha as *const f32) as _,
            input.descriptor,
            input.data,

            self.kernel_descriptor,
            self.kernel,

            self.convolution_descriptor,
            self.algo,

            workspace,
            workspace_size,

            (&self.beta as *const f32) as _,
            output.descriptor,
            output.data,
            self.bias.descriptor,
            self.bias.data,

            self.activation_descriptor,

            output.descriptor,
            output.data,
        ));
      


        match self.activation {
            GpuDnnActivation::Softmax => {

                dnn_error!(cudnnSoftmaxForward(
                    dnn_handle,
                    cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_FAST,
                    cudnnSoftmaxMode_t_CUDNN_SOFTMAX_MODE_CHANNEL,

                    (&1f32 as *const _) as _,
                    output.descriptor,
                    output.data,

                    (&0f32 as *const _) as _,
                    output.descriptor,
                    output.data
                ));

            },
            GpuDnnActivation::LeakyRelu => {
                //NOTE checking workspace size.
                //TODO if workspace is to small maybe allow a realloc? 
                if workspace_size < (4 * output.size()){
                    panic!("Workspace is too small. {} < {}", workspace_size, 4*output.size()); 
                }

                dnn_error!(cudnnAddTensor(
                    dnn_handle,
                    (&-0.3f32 as *const _) as _, 
                    output.descriptor,
                    output.data,
                    (&0f32 as *const _) as _,
                    output.descriptor,
                    workspace,
                )); 
               
                
                dnn_error!(cudnnActivationForward(
                    dnn_handle,
                    self.alt_activation_descriptor, 
                    (&1f32 as *const _) as _,
                    output.descriptor,
                    workspace,
                    (&0f32 as *const _) as _, 
                    output.descriptor,
                    workspace,
                ));
                
                
                dnn_error!(cudnnActivationForward(
                    dnn_handle,
                    self.alt_activation_descriptor, 
                    (&1f32 as *const _) as _,
                    output.descriptor,
                    output.data,
                    (&0f32 as *const _) as _, 
                    output.descriptor,
                    output.data,
                ));
                

                dnn_error!(cudnnAddTensor(
                    dnn_handle,
                    (&-1f32 as *const _) as _, 
                    output.descriptor,
                    workspace,
                    (&1f32 as *const _) as _,
                    output.descriptor,
                    output.data,
                )); 

            },
            _=>{ }
        }

    }
}


pub struct GpuBatchNorm{
    pub epsilon: f32,

    pub descriptor: cudnnTensorDescriptor_t,
    pub scale:    *mut std::ffi::c_void,
    pub bias:     *mut std::ffi::c_void,
    pub est_mean: *mut std::ffi::c_void,
    pub est_var:  *mut std::ffi::c_void,

    pub alpha: f32,
    pub beta:  f32,


}
impl GpuBatchNorm {
    pub fn drop(&mut self){
        dnn_error!(cudnnDestroyTensorDescriptor(self.descriptor));

        cuda_error!(cudaFree(self.scale));
        cuda_error!(cudaFree(self.bias));
        cuda_error!(cudaFree(self.est_mean));
        cuda_error!(cudaFree(self.est_var));
    }

    pub fn construct(channels: usize, epsilon: f32)->Self{
        let mut b = GpuBatchNorm::new();
        b.initialize(channels, epsilon);
        return b;
    }

    pub fn new()->Self{
        GpuBatchNorm{
            epsilon: 0f32,

            descriptor: null_mut(),
            scale:    null_mut(),
            bias:     null_mut(),
            est_mean: null_mut(),
            est_var:  null_mut(),

            alpha: 1f32,
            beta:  0f32,
        }
    }

    pub fn set_scale(&mut self, data: &[u8]){
        let size = data.len();
        cuda_error!(cudaMemcpy(self.scale, data.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    pub fn set_bias(&mut self, data: &[u8]){
        let size = data.len();
        cuda_error!(cudaMemcpy(self.bias, data.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    pub fn set_est_mean(&mut self, data: &[u8]){
        let size = data.len();
        cuda_error!(cudaMemcpy(self.est_mean, data.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    pub fn set_est_var(&mut self, data: &[u8]){
        let size = data.len();
        cuda_error!(cudaMemcpy(self.est_var, data.as_ptr() as _, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }


    pub fn initialize(&mut self, channels: usize, epsilon: f32){

        self.descriptor = {
            let mut descriptor : cudnnTensorDescriptor_t = null_mut();
            dnn_error!(cudnnCreateTensorDescriptor(&mut descriptor as _));
            dnn_error!(cudnnSetTensor4dDescriptor(descriptor,
                                           cudnnTensorFormat_t_CUDNN_TENSOR_NHWC,
                                           cudnnDataType_t_CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                                           channels as _,
                                           1,
                                           1));
            descriptor
        };

        self.epsilon = epsilon;

        cuda_error!(cudaMalloc(&mut self.scale, 4*channels));
        cuda_error!(cudaMalloc(&mut self.bias, 4*channels));
        cuda_error!(cudaMalloc(&mut self.est_mean, 4*channels));
        cuda_error!(cudaMalloc(&mut self.est_var, 4*channels));
    }

    pub fn calc(&self, dnn_handle: cudnnHandle_t, input: &mut GpuTensor, _output: Option<&mut GpuTensor>, ){

        let (output_descriptor, output_data) = match _output {
            Some(op)=>{(op.descriptor, op.data)},
            None=>{(input.descriptor, input.data)},

        };
        dnn_error!(cudnnBatchNormalizationForwardInference(
            dnn_handle,
            cudnnBatchNormMode_t_CUDNN_BATCHNORM_SPATIAL,
            (&self.alpha as *const _) as _, //: *const ::libc::c_void,
            (&self.beta as *const _) as _, //: *const ::libc::c_void,
            input.descriptor, //xDesc: cudnnTensorDescriptor_t,
            input.data,//: *const ::libc::c_void,
            output_descriptor, //yDesc: cudnnTensorDescriptor_t,
            output_data, //y: *mut ::libc::c_void,
            self.descriptor, //bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
            self.scale, //: *const ::libc::c_void,
            self.bias, //: *const ::libc::c_void,
            self.est_mean, //: *const ::libc::c_void,
            self.est_var, //: *const ::libc::c_void,
            self.epsilon as _ ,//: f64,
        ));
    }
}



#[derive(PartialEq, Clone, Copy, Debug)]
pub enum GpuDnnActivation{
    Sigmoid = cudnnActivationMode_t_CUDNN_ACTIVATION_SIGMOID as isize,
    Relu    = cudnnActivationMode_t_CUDNN_ACTIVATION_RELU as isize,
    Tan     = cudnnActivationMode_t_CUDNN_ACTIVATION_TANH as isize,
    ClippedRelu = cudnnActivationMode_t_CUDNN_ACTIVATION_CLIPPED_RELU as isize,
    Elu         = cudnnActivationMode_t_CUDNN_ACTIVATION_ELU as isize,
    Default     = cudnnActivationMode_t_CUDNN_ACTIVATION_IDENTITY as isize,

    //Custom
    LeakyRelu = 998 as isize,
    Softmax   = 999 as isize,
}

pub struct ConvolutionBlock{
    pub conv: GpuConv,
    pub batchnorm: GpuBatchNorm,
    pub leakyrelu: GpuActivationLayer,

    pub output: GpuTensor,
}
impl ConvolutionBlock{
    pub fn drop(&mut self){
        self.conv.drop();
        self.batchnorm.drop();
        self.leakyrelu.drop();
    }
    pub fn new()->Self{
        ConvolutionBlock{
            conv: GpuConv::new(),
            batchnorm: GpuBatchNorm::new(),
            leakyrelu: GpuActivationLayer::new(),

            output: GpuTensor::new(),
        }
    }

    pub fn construct_yolo(kernel: (usize, usize), padding: (i32, i32), strides: (i32, i32), 
                      input_tensor: &GpuTensor, 
                      output_dims: (usize, usize, usize))->Self{
        let mut rt = ConvolutionBlock{
            conv: GpuConv::construct(input_tensor.dims[2], output_dims.2,
                                    kernel.0, kernel.1,

                                    GpuDnnActivation::Default, 
                                    input_tensor.descriptor,

                                    ConvParams{
                                        pad_height: padding.0,
                                        pad_width: padding.1,
                                        vertical_stride: strides.0,
                                        horizontal_stride: strides.1,
                                        dilation_height: 1,
                                        dilation_width: 1,
                                        },
                                    ),
            batchnorm: GpuBatchNorm::construct(output_dims.2, 0.001),
            leakyrelu: GpuActivationLayer::construct(GpuDnnActivation::LeakyRelu),
            output: GpuTensor::new(),
        };

        rt.output.set_descriptor(output_dims.0, output_dims.1, output_dims.2);
        rt.leakyrelu.alpha = -0.1;
        rt
    }

    pub fn calc(&mut self, dnn_handle: cudnnHandle_t, workspace: *mut std::ffi::c_void, workspace_size: usize, input: &GpuTensor){
        self.conv.calc(dnn_handle, workspace, workspace_size, input, &mut self.output);
        self.batchnorm.calc(dnn_handle, &mut self.output, None);
        self.leakyrelu.calc(dnn_handle, workspace, workspace_size, &mut self.output);
    }
}

pub struct GpuPooling{
    pub descriptor: cudnnPoolingDescriptor_t
}


impl GpuPooling {
    pub fn drop(&mut self){
        if self.descriptor == null_mut() {
            return;
        }
        dnn_error!(cudnnDestroyPoolingDescriptor(
            self.descriptor,
        ));
        
    }
    pub fn new()->Self{
        GpuPooling{
            descriptor: null_mut(),
        }
    }

    pub fn initialize(&mut self, width: i32, height: i32, vertical_stride: i32, horizontal_stride: i32){
        dnn_error!(cudnnCreatePoolingDescriptor(
            &mut self.descriptor,
        ));
        dnn_error!(cudnnSetPooling2dDescriptor(
            self.descriptor,
            cudnnPoolingMode_t_CUDNN_POOLING_MAX, //mode: cudnnPoolingMode_t,
            cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN, //maxpoolingNanOpt: cudnnNanPropagation_t,
            height, //windowHeight: ::libc::c_int,
            width, //windowWidth: ::libc::c_int,
            0, //verticalPadding: ::libc::c_int,
            0, //horizontalPadding: ::libc::c_int,
            vertical_stride, //verticalStride: ::libc::c_int,   
            horizontal_stride, //horizontalStride: ::libc::c_int,
        ));
    }

    pub fn calc(&self, handle: cudnnHandle_t, input: &GpuTensor, output: &mut GpuTensor){
        dnn_error!(cudnnPoolingForward(
            handle,
            self.descriptor,
            (&1f32 as *const f32) as _, //alpha: *const ::libc::c_void,
            input.descriptor, //xDesc: cudnnTensorDescriptor_t,
            input.data, //x: *const ::libc::c_void,
            (&0f32 as *const f32) as _, //beta: *const ::libc::c_void,
            output.descriptor, //yDesc: cudnnTensorDescriptor_t,
            output.data, //y: *mut ::libc::c_void,
        ));
    }
}


pub struct CspBlock{
    _input_split_descriptor: cudnnTensorDescriptor_t,
    _intermediate_concat_descriptor2: cudnnTensorDescriptor_t,

    pub input_split_tensor: GpuTensor,
    pub c1: ConvolutionBlock,
    pub c2: ConvolutionBlock,
    pub c3: ConvolutionBlock,
    pub concat_c1_c2: GpuTensor,
    pub output: GpuTensor,
}

impl CspBlock{
    pub fn drop(&mut self){
        dnn_error!(cudnnDestroyTensorDescriptor(self._input_split_descriptor));
        dnn_error!(cudnnDestroyTensorDescriptor(self._intermediate_concat_descriptor2));

        self.input_split_tensor.drop();
        self.c1.drop();
        self.c2.drop();
        self.c3.drop();
        self.concat_c1_c2.drop();
        self.output.drop();

    }

    pub fn construct_yolo(input_tensor: &GpuTensor, output_shape: (usize, usize, usize))->Self{

        let concat_c1_c2 = GpuTensor::construct(output_shape.0, output_shape.1, output_shape.2/2);
        let input_split_tensor = GpuTensor::construct(output_shape.0, output_shape.1, input_tensor.dims[2]/2);

        let c1 =  ConvolutionBlock::construct_yolo((3, 3), 
                                                   (1, 1), 
                                                   (1, 1), 
                                                   &input_split_tensor, //<- should be the split tensor 
                                                   (output_shape.0, output_shape.1, input_tensor.dims[2]/2));
        let mut rt = CspBlock{
            _input_split_descriptor: null_mut(),
            _intermediate_concat_descriptor2: null_mut(),

            input_split_tensor: input_split_tensor,

            c1: ConvolutionBlock::new(),
            c2: ConvolutionBlock::construct_yolo((3,3), (1, 1), (1, 1), 
                                                  &c1.output, 
                                                  (output_shape.0, output_shape.1, input_tensor.dims[2]/2)),
            c3: ConvolutionBlock::construct_yolo((1, 1), (0, 0), (1, 1),
                                                  &concat_c1_c2, 
                                                  (output_shape.0, output_shape.1, input_tensor.dims[2])),
            concat_c1_c2: concat_c1_c2,
            output: GpuTensor::construct(output_shape.0, output_shape.1, output_shape.2),
        };

        dnn_error!(cudnnCreateTensorDescriptor(&mut rt._input_split_descriptor));
        {
            let _c = input_tensor.dims[2]/2;
            let _h = output_shape.1;
            let _w = output_shape.0;

            dnn_error!(cudnnSetTensor4dDescriptorEx(
                rt._input_split_descriptor, 
                cudnnDataType_t_CUDNN_DATA_FLOAT,
                1,
                _c as _,
                _h as _,
                _w as _,
                (2 * _c * _h * _w) as _,
                1,
                (2 * _c * _w) as _,
                (2 * _c ) as _,
            ));
        }

        dnn_error!(cudnnCreateTensorDescriptor(&mut rt._intermediate_concat_descriptor2));
        {
            let _c = input_tensor.dims[2];
            let _h = output_shape.1;
            let _w = output_shape.0;

            dnn_error!(cudnnSetTensor4dDescriptorEx(
                rt._intermediate_concat_descriptor2, 
                cudnnDataType_t_CUDNN_DATA_FLOAT,
                1,
                _c as _,
                _h as _,
                _w as _,
                (2 * _c * _h * _w) as _,
                1,
                //(2 * _c) as _,
                (2 * _c * _w) as _,
                (2 * _c) as _
            ));
        }



        rt.c1 = c1;

        rt.c1.output.gpu_alloc(rt.c1.output.size_bytes());
        rt.c2.output.gpu_alloc(rt.c2.output.size_bytes());
        rt.c3.output.gpu_alloc(rt.c3.output.size_bytes());


        rt
    }

    pub fn load_weights(&mut self, c1: [&Layer; 2], c2: [&Layer; 2], c3: [&Layer; 2]){

        match c1[0]._type {
            LayerType::Conv2D(_)=>{
            },
            _=>{panic!("c1: expected a Conv2d layer");}
        }
        
        self.c1.conv.set_kernel(&c1[0].arrays[0].data);

        match c1[1]._type {
            LayerType::BatchNorm(_)=>{},
            _=>{panic!("c1: expected a Conv2d layer");}
        }
        self.c1.batchnorm.set_scale(   &c1[1].arrays[0].data);
        self.c1.batchnorm.set_bias(    &c1[1].arrays[1].data);
        self.c1.batchnorm.set_est_mean(&c1[1].arrays[2].data);
        self.c1.batchnorm.set_est_var( &c1[1].arrays[3].data);


        match c2[0]._type {
            LayerType::Conv2D(_)=>{},
            _=>{panic!("c1: expected a Conv2d layer");}
        }
        self.c2.conv.set_kernel(&c2[0].arrays[0].data);

        match c2[1]._type {
            LayerType::BatchNorm(_)=>{},
            _=>{panic!("c1: expected a Conv2d layer");}
        }
        self.c2.batchnorm.set_scale(   &c2[1].arrays[0].data);
        self.c2.batchnorm.set_bias(    &c2[1].arrays[1].data);
        self.c2.batchnorm.set_est_mean(&c2[1].arrays[2].data);
        self.c2.batchnorm.set_est_var( &c2[1].arrays[3].data);


        match c3[0]._type {
            LayerType::Conv2D(_)=>{},
            _=>{panic!("c1: expected a Conv2d layer");}
        }
        self.c3.conv.set_kernel(&c3[0].arrays[0].data);


        match c3[1]._type {
            LayerType::BatchNorm(_)=>{},
            _=>{panic!("c1: expected a Conv2d layer");}
        }
        self.c3.batchnorm.set_scale(   &c3[1].arrays[0].data);
        self.c3.batchnorm.set_bias(    &c3[1].arrays[1].data);
        self.c3.batchnorm.set_est_mean(&c3[1].arrays[2].data);
        self.c3.batchnorm.set_est_var( &c3[1].arrays[3].data);
    
    }

    pub fn calc(&mut self, dnn_handle: cudnnHandle_t, workspace: *mut std::ffi::c_void, workspace_size: usize, input: &GpuTensor){

        //Step 1: Split data
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          self._input_split_descriptor, //xDesc: cudnnTensorDescriptor_t,
          input.data.offset((4*input.dims[2]/2) as _), //x: *const ::libc::c_void,
          (&0f32 as *const _) as *const _,
          self.input_split_tensor.descriptor, //yDesc: cudnnTensorDescriptor_t,
          self.input_split_tensor.data, //y: *mut ::libc::c_void,
        ));


        //Step 2: calc
        self.c1.calc(dnn_handle, workspace, workspace_size, &self.input_split_tensor);
        self.c2.calc(dnn_handle, workspace, workspace_size, &mut self.c1.output);

        //Step 3: Conat c1 and c2 outputs 
        concat_tensor(dnn_handle, &self.c2.output, &self.c1.output, &mut self.concat_c1_c2, self._input_split_descriptor, self._input_split_descriptor);

        //Step 4: calc
        self.c3.calc(dnn_handle, workspace, workspace_size, &mut self.concat_c1_c2);

        //Step 5: concat orig inputs with c3 output
        concat_tensor(dnn_handle, input, &self.c3.output, &mut self.output, self._intermediate_concat_descriptor2, self._intermediate_concat_descriptor2);
    }
}


pub fn concat_tensor(dnn_handle: cudnnHandle_t, input1: &GpuTensor, input2: &GpuTensor, output: &mut GpuTensor, intermediate_descriptor1: cudnnTensorDescriptor_t, intermediate_descriptor2: cudnnTensorDescriptor_t){

    if input1.dims[2] + input2.dims[2] != output.dims[2] {
        panic!("Dimensions are not correct input1:{:?}, input2:{:?}, output:{:?}", input1.dims, input2.dims, output.dims);
    }

    dnn_error!(cudnnTransformTensor(
      dnn_handle,
      (&1f32 as *const _) as *const _,
      input1.descriptor,
      input1.data, 
      (&0f32 as *const _) as *const _,
      intermediate_descriptor1,
      output.data,
    ));

    dnn_error!(cudnnTransformTensor(
      dnn_handle,
      (&1f32 as *const _) as *const _,
      input2.descriptor,
      input2.data,
      (&0f32 as *const _) as *const _,
      intermediate_descriptor2,
      output.data.offset((input1.dims[2]*4) as _),
    ));

}

