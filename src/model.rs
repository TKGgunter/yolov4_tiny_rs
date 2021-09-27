use crate::numpy::*;
use crate::dnn::*;
use crate::cuda11_cudnn_sys::{cudnnActivationMode_t_CUDNN_ACTIVATION_SIGMOID,
                              cudnnActivationMode_t_CUDNN_ACTIVATION_RELU,
                              cudnnActivationMode_t_CUDNN_ACTIVATION_TANH,
                              cudnnActivationMode_t_CUDNN_ACTIVATION_IDENTITY,
                              cudnnActivationMode_t_CUDNN_ACTIVATION_CLIPPED_RELU,
                              cudnnActivationMode_t_CUDNN_ACTIVATION_ELU,
                             };
use crate::cuda11_cublasLt_sys::{cublasLtEpilogue_t_CUBLASLT_EPILOGUE_DEFAULT,
                                 cublasLtEpilogue_t_CUBLASLT_EPILOGUE_RELU,
                                 cublasLtEpilogue_t_CUBLASLT_EPILOGUE_BIAS,
                                 cublasLtEpilogue_t_CUBLASLT_EPILOGUE_RELU_BIAS,
                                };
use crate::_break_loop;

use std::mem::size_of;
use std::mem::transmute;
use std::fs::File;
use std::io::prelude::*;

#[repr(i32)]
pub enum LayerTypei32{
    Dense        = 0,
    Conv2D       = 1,
    MaxPooling2D = 2,
    Flatten      = 3,
    BatchNorm    = 4,
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

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum GpuMatrixEpilogueSettings{
    Default = cublasLtEpilogue_t_CUBLASLT_EPILOGUE_DEFAULT as isize,
    Relu    = cublasLtEpilogue_t_CUBLASLT_EPILOGUE_RELU as isize,
    Bias    = cublasLtEpilogue_t_CUBLASLT_EPILOGUE_BIAS as isize,
    BiasRelu = cublasLtEpilogue_t_CUBLASLT_EPILOGUE_RELU_BIAS as isize,

    //Custom
    LeakyRelu = 998 as isize,
    Softmax   = 999 as isize,
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct BatchNormConfig{ 
    epsilon: f32 
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct DenseConfig{ 
    activation: i32 
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Conv2DConfig{ 
    pub batch_input_shape: [i32; 3], 
    pub kernel_size: [i32; 2], 
    pub strides: [i32; 2], 
    pub filters: i32,
    pub activation: i32 
//TODO adding will need to be added in the future
}

pub fn convert_activation_conv(x: i32)->GpuDnnActivation{
    match x {
        0 => { return GpuDnnActivation::Default; },
        1 => { return GpuDnnActivation::Relu;},
        998 => { return GpuDnnActivation::LeakyRelu;},
        999 => { return GpuDnnActivation::Softmax;},
        _ => panic!("activation is not handled")
    }
}

pub fn convert_activation_dense(x: i32)->GpuMatrixEpilogueSettings{
    match x {
        0 => { return GpuMatrixEpilogueSettings::Default; },
        1 => { return GpuMatrixEpilogueSettings::Relu;},
        998 => { return GpuMatrixEpilogueSettings::LeakyRelu;},
        999 => { return GpuMatrixEpilogueSettings::Softmax;},
        _ => panic!("activation is not handled")
    }
}


#[derive(Debug, Default)]
#[repr(C)]
pub struct MaxPooling2DConfig{ 
    pub pool_size: [i32; 2], 
    pub strides: [i32; 2], 
//TODO adding will need to be added in the future
}

#[derive(Debug)]
pub enum LayerType{
    DenseLayer(DenseConfig), //TODO will need it's own configs to handle activation
    Conv2D(Conv2DConfig),
    MaxPooling2D(MaxPooling2DConfig),
    FlattenLayer,
    BatchNorm(BatchNormConfig),
    Default,
}
pub struct Layer{
    pub _type: LayerType,
    pub arrays: Vec<NumpyArray>,
    pub output_dims: Vec<i32>
}
impl Layer{
    fn new()->Self{
        Layer{
            _type: LayerType::Default,
            arrays: Vec::new(),
            output_dims: Vec::new(),
        }
    }
}
pub fn load_layers(file_name: &str)->Vec<Layer>{
    //LOAD FILE handle
    let mut f = File::open(file_name).expect("Could not open file.");



    let mut rt = vec![];
    loop/*still more bytes to read*/{unsafe{

        let mut layer = Layer::new();
        //load type



        /////////////////////////////////////
        macro_rules! match_layertype_configs{
            ($enum_type:expr, $config_type:ty) =>{
                let mut config = <$config_type>::default();

                {
                    let _config = transmute::<_, &mut [u8; size_of::<$config_type>()]>(&mut config);
                    _break_loop!( f.read_exact(_config) );
                }
                layer._type = $enum_type(config);
            }
        }
        /////////////////////////////////////



        let mut layer_type = LayerTypei32::Dense;
        {
            let _layer_type = transmute::<_, &mut [u8; 4]>(&mut layer_type);
            _break_loop!( f.read_exact(_layer_type) );
        }

        //NOTE
        //Get output dimns
        {
            let mut dims_size = 0i32;
            _break_loop!( f.read_exact(transmute::<_, &mut [u8; 4]>(&mut dims_size)) );

            let mut dims = vec![0u8; 4*dims_size as usize];
            _break_loop!( f.read_exact(dims.as_mut_slice()) );


            //TODO I don't really like allocing a new vector. 
            //I would rather convert the u8 dims to i32, do I don't think the lang can do this.
            layer.output_dims.extend_from_slice(&transmute::<_, &[i32]>(&dims[..])[..dims_size as _]);
        }


        match layer_type {
            LayerTypei32::Dense  =>{
                match_layertype_configs!(LayerType::DenseLayer, DenseConfig);
            },
            LayerTypei32::Conv2D =>{ 
                match_layertype_configs!(LayerType::Conv2D, Conv2DConfig);
            },
            LayerTypei32::MaxPooling2D =>{
                match_layertype_configs!(LayerType::MaxPooling2D, MaxPooling2DConfig);
            }, 
            LayerTypei32::Flatten => layer._type = LayerType::FlattenLayer,
            LayerTypei32::BatchNorm =>{
                match_layertype_configs!(LayerType::BatchNorm, BatchNormConfig);
            },
            _ =>panic!("unexpected layertype from save file."),
        }

        let mut n_arrays = 0i32;
        {
            let _n_arrays = transmute::<_, &mut [u8; 4]>(&mut n_arrays);
            _break_loop!( f.read_exact(_n_arrays) );
        }

        for _i in 0..n_arrays {
            let mut array = NumpyArray::new();
            match array.from_file(&mut f) {
                Ok(())=>(),
                Err(())=>break,
            }
            layer.arrays.push(array)
        }
        rt.push(layer)
    }}

    return rt;
}

