#![allow(dead_code)]

extern crate stb_tt_sys;
extern crate stb_image_write_sys;
extern crate stb_image_sys;
extern crate stb_resize_sys;
extern crate cuda11_cudnn_sys;
extern crate cuda11_cutensor_sys;
extern crate cuda11_cublasLt_sys;
extern crate clap;

mod numpy;
mod dnn;
mod model;
mod tensor;

use std::fs::File;
use std::io::prelude::*;
use std::ptr::null_mut;

use stb_image_sys::stbi_load_from_memory_32bit;
use stb_resize_sys::stbir_resize_float;
use stb_image_write_sys::write_png;
use crate::stb_tt_sys::*;

use cuda11_cudnn_sys::*;
use cuda11_cutensor_sys::{cutensorGetErrorString, cutensorInit, cutensorHandle_t, };

use numpy::NumpyArray;
use dnn::*;
use dnn::cudaMalloc;
use model::*;
use tensor::*;
use clap::{Arg, App};



#[macro_use]
macro_rules! tensor_error{
    ($x:expr)=>{
        let x = unsafe{ $x };
        if x != 0 {unsafe{
            let str = cutensorGetErrorString(x);
            panic!("Tensor error: {} {:?}", x, std::ffi::CStr::from_ptr(str));
        }}
    }
}
pub(crate) use tensor_error;

#[macro_use]
macro_rules! dnn_error{
    ($x:expr) => {
        let y = unsafe{ $x };
        if y != cudnnStatus_t_CUDNN_STATUS_SUCCESS{unsafe{
            let str = cudnnGetErrorString(y);
            panic!("Error dnn: {:?} {:?}", y, std::ffi::CStr::from_ptr(str));
        }}
    }
}
pub(crate) use dnn_error;

#[macro_use]
macro_rules! cuda_error{
    ($x:expr) => {
        let y = unsafe{ $x };
        if y != cudaError::cudaSuccess {unsafe{
            let str = cudaGetErrorString(y);
            panic!("Error cuda: {:?} {:?}", y, std::ffi::CStr::from_ptr(str));
        }}
    }
}
pub(crate) use cuda_error;

#[macro_use]
macro_rules! _break_loop {
    ($x:expr)=>{
        match $x {
            Ok(_)=>{},
            _=>{break;}
        }
    }
}
pub(crate) use _break_loop;



fn main() {

    let opts = App::new("yolov4-tiny-rs")
                     .version("1.0")
                     .author("Thoth G. <thothgunter@live.com>")
                     .about("Applies the yolo algorithm to provided image and produces an image with the according labels and bounding boxes.")
                     .arg(Arg::with_name("INPUT")
                          .help("Sets the input file.")
                          .required(true)
                          .index(1))
                     .arg(Arg::with_name("output")
                          .short("o")
                          .long("output")
                          .value_name("FILE")
                          .help("Sets the output file's name.")
                          .takes_value(true))
                     .get_matches();



    //FUTURE
    //TODO let user define output type with post fix.
    let default_out_file_name = "out.png";


    let input_file  = opts.value_of("INPUT").unwrap();
    let mut output_file = default_out_file_name;

    match opts.value_of("output").as_ref() {
        Some(x) => { output_file = x; },
        None => { 
            println!("Output file name was not set. Default output file name: {:?}", default_out_file_name); 
        }
    }



    //TODO
    //load file
    //set up output dimensions
    let mut buffer_f32 = vec![0f32; 416*416*3];
    let org_img_width;
    let org_img_height;
    let mut org_img = Image{w: 0, h: 0, buffer: Vec::new()};

    {//Load and prep input data
        println!("Loading: {}", input_file);

        let mut file = File::open(input_file).expect("Could not find input file.");
        let mut file_buffer = vec![];

        file.read_to_end(&mut file_buffer).expect("Could not read to end of file.");
        let img = stbi_load_from_memory_32bit(&file_buffer).expect("could not load image");

        //TODO redundant
        org_img.w = img.width; 
        org_img.h = img.height;
        org_img_width = img.width;
        org_img_height = img.height;


        let mut i = 0;
        let mut _buffer_f32 = vec![0f32; img.buffer.len()];
        for _i in 0.._buffer_f32.len() {
            
            if (_i+1) % 4 == 0 { continue; }

            _buffer_f32[i] = img.buffer[_i] as f32 / 255f32;
            org_img.buffer.push(img.buffer[_i]);

            i += 1;
        }


        let error = unsafe{
                      stbir_resize_float( _buffer_f32.as_ptr(), img.width, img.height, 
                            4*img.width*3, buffer_f32.as_mut_ptr(),
                            416, 416, 4*416*3,
                            3)
                    };
        if error == 0 {
            panic!("Image could not be resized.");
        }
    }

    //NOTE
    //setup cuda
    let mut cuda_stream = null_mut();
    cuda_error!(cudaStreamCreate(&mut cuda_stream));

    let mut workspace = null_mut();
    let workspace_size = 18877456; 
    cuda_error!(cudaMalloc(&mut workspace as _, workspace_size));


    let mut dnn_handle = null_mut();
    dnn_error!(cudnnCreate(&mut dnn_handle));

    
    let mut cutensor_handle = cutensorHandle_t {
        fields: [0i64; 512usize],
    };
    tensor_error!(cutensorInit(&mut cutensor_handle));


    //NOTE
    //setup yolov4
    let layers = load_layers("data/yolov4_tiny_tg");

    let mut input_tensor = GpuTensor::new();
    input_tensor.initialize(416, 416, 3);
    input_tensor.set_data(&buffer_f32);
    
    let mut b1 = ConvolutionBlock::construct_yolo((3, 3),  //Kernel
                                                  (1, 1),  //Padding
                                                  (2, 2),  //Strides
                                                  &input_tensor, (208, 208, 32));
    b1.output.gpu_alloc(b1.output.size_bytes());
    b1.conv.set_kernel(&layers[0].arrays[0].data);
    b1.conv.bias.fill_with_scalar(dnn_handle, 0f32);
    b1.conv.beta = 0f32;
    b1.conv.alpha = 1.0f32;

    b1.batchnorm.set_scale(&layers[1].arrays[0].data);
    b1.batchnorm.set_bias(&layers[1].arrays[1].data);
    b1.batchnorm.set_est_mean(&layers[1].arrays[2].data);
    b1.batchnorm.set_est_var(&layers[1].arrays[3].data);


    let mut b2 = ConvolutionBlock::construct_yolo((3, 3),  //Kernel
                                                  (1, 1),  //Padding
                                                  (2, 2),  //Strides
                                                  &b1.output, (104, 104, 64));

    b2.output.gpu_alloc(b2.output.size_bytes()); //TODO we should be able to reuse the output of b1
    b2.conv.set_kernel(&layers[2].arrays[0].data);


    b2.batchnorm.set_scale(&layers[3].arrays[0].data);
    b2.batchnorm.set_bias(&layers[3].arrays[1].data);
    b2.batchnorm.set_est_mean(&layers[3].arrays[2].data);
    b2.batchnorm.set_est_var(&layers[3].arrays[3].data);
    

    let mut b3 = ConvolutionBlock::construct_yolo((3, 3),  //Kernel
                                                  (1, 1),  //Padding
                                                  (1, 1),  //Strides
                                                  &b2.output, (104, 104, 64));

    b3.output.gpu_alloc(b3.output.size_bytes());  //TODO we should be able to reuse the output of b1 and b2
    b3.conv.set_kernel(&layers[4].arrays[0].data);

    b3.batchnorm.set_scale(&layers[5].arrays[0].data);
    b3.batchnorm.set_bias(&layers[5].arrays[1].data);
    b3.batchnorm.set_est_mean(&layers[5].arrays[2].data);
    b3.batchnorm.set_est_var(&layers[5].arrays[3].data);


    let mut csp1 = CspBlock::construct_yolo(&b3.output, (104, 104, 128)); 
    csp1.load_weights([&layers[6], &layers[7]], [&layers[8], &layers[9]], [&layers[10], &layers[11]]);



    let mut maxpooling1 = GpuPooling::new();
    maxpooling1.initialize(2, 2, 2, 2);

    let mut maxpooling1_output = GpuTensor::construct(52, 52, 128);
    maxpooling1_output.gpu_alloc(maxpooling1_output.size_bytes());


    let mut b4 = ConvolutionBlock::construct_yolo((3, 3), 
                                                  (1, 1), 
                                                  (1, 1), 
                                                  &maxpooling1_output, (52, 52, 128));
    b4.output.gpu_alloc(b4.output.size_bytes());  
    b4.conv.set_kernel(&layers[13].arrays[0].data);
    b4.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    b4.batchnorm.set_scale(&layers[14].arrays[0].data);
    b4.batchnorm.set_bias( &layers[14].arrays[1].data);
    b4.batchnorm.set_est_mean(&layers[14].arrays[2].data);
    b4.batchnorm.set_est_var(&layers[14].arrays[3].data);


    let mut csp2 = CspBlock::construct_yolo(&b4.output, (52, 52, 256)); 
    csp2.load_weights([&layers[15], &layers[16]], [&layers[17], &layers[18]], [&layers[19], &layers[20]]);


    let mut maxpooling2_output = GpuTensor::construct(26, 26, 256);
    maxpooling2_output.gpu_alloc(maxpooling2_output.size_bytes());

    let mut b5 = ConvolutionBlock::construct_yolo((3, 3), 
                                                  (1, 1), 
                                                  (1, 1), 
                                                  &maxpooling2_output, (26, 26, 256));

    b5.output.gpu_alloc(b5.output.size_bytes());  
    b5.conv.set_kernel(&layers[22].arrays[0].data);
    b5.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    b5.batchnorm.set_scale(&layers[23].arrays[0].data);
    b5.batchnorm.set_bias( &layers[23].arrays[1].data);
    b5.batchnorm.set_est_mean(&layers[23].arrays[2].data);
    b5.batchnorm.set_est_var(&layers[23].arrays[3].data);
    

    let mut csp3 = CspBlock::construct_yolo(&b5.output, (26, 26, 512)); 
    csp3.load_weights([&layers[24], &layers[25]], [&layers[26], &layers[27]], [&layers[28], &layers[29]]);

    let mut maxpooling3_output = GpuTensor::construct(13, 13, 512);
    maxpooling3_output.gpu_alloc(maxpooling3_output.size_bytes());

    let mut b6 = ConvolutionBlock::construct_yolo((3, 3), 
                                                  (1, 1), 
                                                  (1, 1), 
                                                  &maxpooling3_output, (13, 13, 512));

    b6.output.gpu_alloc(b6.output.size_bytes());  
    b6.conv.set_kernel(&layers[31].arrays[0].data);
    b6.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    b6.batchnorm.set_scale(&layers[32].arrays[0].data);
    b6.batchnorm.set_bias( &layers[32].arrays[1].data);
    b6.batchnorm.set_est_mean(&layers[32].arrays[2].data);
    b6.batchnorm.set_est_var(&layers[32].arrays[3].data);

    ////////////////////////////
    //branch lower (no upsample)
    let mut l_b1 = ConvolutionBlock::construct_yolo((1, 1), 
                                                    (0, 0), 
                                                    (1, 1), 
                                                    &b6.output, (13, 13, 256));

    l_b1.output.gpu_alloc(l_b1.output.size_bytes());  
    l_b1.conv.set_kernel(&layers[33].arrays[0].data);
    l_b1.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    l_b1.batchnorm.set_scale(&layers[34].arrays[0].data);
    l_b1.batchnorm.set_bias( &layers[34].arrays[1].data);
    l_b1.batchnorm.set_est_mean(&layers[34].arrays[2].data);
    l_b1.batchnorm.set_est_var(&layers[34].arrays[3].data);



    let mut l_b2 = ConvolutionBlock::construct_yolo((3, 3), 
                                                  (1, 1), 
                                                  (1, 1), 
                                                  &l_b1.output, (13, 13, 512));

    l_b2.output.gpu_alloc(b6.output.size_bytes());  
    l_b2.conv.set_kernel(&layers[38].arrays[0].data);
    l_b2.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    l_b2.batchnorm.set_scale(&layers[40].arrays[0].data);
    l_b2.batchnorm.set_bias( &layers[40].arrays[1].data);
    l_b2.batchnorm.set_est_mean(&layers[40].arrays[2].data);
    l_b2.batchnorm.set_est_var(&layers[40].arrays[3].data);


    let number_classes = 80;
    let mut bbox = GpuConv::construct(l_b2.output.dims[2], (number_classes + 5) * 3,
                                     1, 1,
                                     GpuDnnActivation::Default, 
                                     l_b2.output.descriptor,
                                     ConvParams{
                                          pad_height: 0,
                                          pad_width : 0,
                                          vertical_stride  : 1,
                                          horizontal_stride: 1,
                                          dilation_height: 1,
                                          dilation_width : 1,
                                     });

    bbox.set_kernel(&layers[42].arrays[0].data);
    bbox.set_bias(&layers[42].arrays[1].data);

    let mut bbox_output = GpuTensor::construct(13, 13, (number_classes + 5) * 3);

    /////////////////////////////////////
    //branch upper (with upsample)
    let mut u_b1 = ConvolutionBlock::construct_yolo((1, 1), 
                                                    (0, 0), 
                                                    (1, 1), 
                                                    &l_b1.output, (13, 13, 128));

    u_b1.output.gpu_alloc(u_b1.output.size_bytes());  
    u_b1.conv.set_kernel(&layers[35].arrays[0].data);
    u_b1.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    u_b1.batchnorm.set_scale(&layers[36].arrays[0].data);
    u_b1.batchnorm.set_bias( &layers[36].arrays[1].data);
    u_b1.batchnorm.set_est_mean(&layers[36].arrays[2].data);
    u_b1.batchnorm.set_est_var(&layers[36].arrays[3].data);


    let mut resized = [0f32; 26*26*128];
    let mut resized_gpu = GpuTensor::construct(26, 26, 128);

    let mut u_concat_tensor = GpuTensor::construct(26, 26, 384);

    let mut u_inter_concat_descriptor1 = null_mut();
    let mut u_inter_concat_descriptor2 = null_mut();
    dnn_error!(cudnnCreateTensorDescriptor(&mut u_inter_concat_descriptor1));
    dnn_error!(cudnnCreateTensorDescriptor(&mut u_inter_concat_descriptor2));
    {
        let _c = 128;
        let _h = 26;
        let _w = 26;

        dnn_error!(cudnnSetTensor4dDescriptorEx(
            u_inter_concat_descriptor1, 
            cudnnDataType_t_CUDNN_DATA_FLOAT,
            1,
            _c as _,
            _h as _,
            _w as _,
            (384 * _h * _w) as _,
            1,
            (384 * _w) as _,
            (384 ) as _,
        ));

        let _c = 256;
        dnn_error!(cudnnSetTensor4dDescriptorEx(
            u_inter_concat_descriptor2, 
            cudnnDataType_t_CUDNN_DATA_FLOAT,
            1,
            _c as _,
            _h as _,
            _w as _,
            (384 * _h * _w) as _,
            1,
            (384 * _w) as _,
            (384 ) as _,
        ));
    }

    let mut u_b2 = ConvolutionBlock::construct_yolo((3, 3), 
                                                    (1, 1), 
                                                    (1, 1), 
                                                    &u_concat_tensor, (26, 26, 256));

    u_b2.output.gpu_alloc(u_b2.output.size_bytes());  
    u_b2.conv.set_kernel(&layers[37].arrays[0].data);
    u_b2.conv.bias.fill_with_scalar(dnn_handle, 0f32);

    u_b2.batchnorm.set_scale(&layers[39].arrays[0].data);
    u_b2.batchnorm.set_bias( &layers[39].arrays[1].data);
    u_b2.batchnorm.set_est_mean(&layers[39].arrays[2].data);
    u_b2.batchnorm.set_est_var(&layers[39].arrays[3].data);

    let mut mbbox = GpuConv::construct(u_b2.output.dims[2], (number_classes + 5) * 3,
                                     1, 1,
                                     GpuDnnActivation::Default, 
                                     u_b2.output.descriptor,
                                     ConvParams{
                                          pad_height: 0,
                                          pad_width : 0,
                                          vertical_stride  : 1,
                                          horizontal_stride: 1,
                                          dilation_height: 1,
                                          dilation_width : 1,
                                     });


    mbbox.set_kernel(&layers[41].arrays[0].data);
    mbbox.set_bias(&layers[41].arrays[1].data);

    let mut mbbox_output = GpuTensor::construct(26, 26, (number_classes + 5) * 3);


    /////////////////////////////////////
    //decoding block

    fn construct_ndtensor_descriptors( dims: &[i32], strides: &[i32])->cudnnTensorDescriptor_t{

        let mut rt : cudnnTensorDescriptor_t = null_mut();
        dnn_error!(cudnnCreateTensorDescriptor(&mut rt as _));

        dnn_error!(cudnnSetTensorNdDescriptor(
            rt,                               //tensorDesc: cudnnTensorDescriptor_t,
            cudnnDataType_t_CUDNN_DATA_FLOAT, //dataType: cudnnDataType_t,
            dims.len() as _,                  //nbDims: ::libc::c_int,
            dims.as_ptr(),                    //dimA: *const ::libc::c_int,
            strides.as_ptr(),
        )); 

        rt
    }

    //////////////////
    //13x13 block
    let dim_labels = ['a', 'b', 'c', 'd'];
    let _nc = number_classes as i32;

    let nxn = 13;

    let org_strides13 = [nxn*3*(_nc+5), 3*(_nc+5), (_nc+5), 1,];

    let dims    = [nxn, nxn, 3, 2];
    let strides = [2*3*nxn, 2*3, 2, 1];

    let dxdy_desc13 = construct_ndtensor_descriptors(&dims, &strides);
    let from_dxdy_desc13 = construct_ndtensor_descriptors(&dims, &org_strides13);
    

    let mut dxdy_ttensor_13 = GpuTensorTensor::construct( &cutensor_handle, &dims, GpuTensorOps::Sigmoid);
    let mut dxdy_ttensor_ones_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Identity);
    let mut dxdy_ttensor_out_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Identity);


    dnn_error!(cudnnSetTensor(
            dnn_handle,
            dxdy_desc13,
            dxdy_ttensor_ones_13.data,
            (&(-0.5f32*(1.05-1.0)) as *const f32) as _,
    ));



    let mut dwdh_ttensor_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Exp);


    let mut cpu_dwdh_anchors_13 = NumpyArray::new();
    cpu_dwdh_anchors_13.from_file(&mut File::open("data/anchors_1").expect("anchors_1")).expect("Could not get anchors from file.");


    let mut anchors_ttensor_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Identity);

    cuda_error!(cudaMemcpy(anchors_ttensor_13.data, cpu_dwdh_anchors_13.data.as_ptr() as _, 4*3*2,
                          cudaMemcpyKind::cudaMemcpyHostToDevice));

    //TODO 09/17/21 I have yet to figure out how one broadcast tensors to a higher dimension
    //We are doing this instead.
    for i in 1..nxn*nxn {
        cuda_error!(cudaMemcpy(anchors_ttensor_13.data.offset((i*3*2*4) as _), anchors_ttensor_13.data, 4*3*2,
                              cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }
    



    let dims    = [nxn,nxn,3,1];
    let strides = [3*nxn*_nc,3*_nc, _nc, 1];
    let from_conf_desc13 = construct_ndtensor_descriptors(&dims, &org_strides13);
    let conf_desc13 = construct_ndtensor_descriptors(&dims, &strides);




    let dims = [nxn,nxn,3,_nc];
    let strides = [_nc*nxn*3, _nc*3,_nc,1];

    let mut conf_ttensor_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Sigmoid);
    let mut prob_ttensor_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Sigmoid);

    let from_prob_desc13 = construct_ndtensor_descriptors(&dims, &org_strides13);
    let prob_desc13 = construct_ndtensor_descriptors(&dims, &strides);


    let mut prob_ttensor_out_13 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Sigmoid);


    let mut cpu_xygrid_13 = NumpyArray::new();
    cpu_xygrid_13.from_file(&mut File::open("data/xygrid_13").expect("xygrid_13")).expect("Could not get xy grids from file.");

    let mut xygrid_13_data : *mut std::ffi::c_void = null_mut(); 
    cuda_error!(cudaMalloc(&mut xygrid_13_data, (4*nxn*nxn*3*2) as _));
    cuda_error!(cudaMemcpy(xygrid_13_data,  cpu_xygrid_13.data.as_ptr() as _, (nxn*nxn*3*2*4) as _, 
                           cudaMemcpyKind::cudaMemcpyHostToDevice));


    //////////////////
    //26x26 block

    let nxn = 26;

    let org_strides26 = [nxn*3*(_nc+5), 3*(_nc+5), (_nc+5), 1,];

    let dims    = [nxn, nxn, 3, 2];
    let strides = [2*3*nxn, 2*3, 2, 1];

    let dxdy_desc26 = construct_ndtensor_descriptors(&dims, &strides);
    let from_dxdy_desc26 = construct_ndtensor_descriptors(&dims, &org_strides26);
    

    let mut dxdy_ttensor_26 = GpuTensorTensor::construct( &cutensor_handle, &dims, GpuTensorOps::Sigmoid);
    let mut dxdy_ttensor_ones_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Identity);
    let mut dxdy_ttensor_out_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Identity);


    dnn_error!(cudnnSetTensor(
            dnn_handle,
            dxdy_desc26,
            dxdy_ttensor_ones_26.data,
            (&(-0.5f32*(1.05-1.0)) as *const f32) as _,
    ));

    let mut dwdh_ttensor_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Exp);


    let mut cpu_dwdh_anchors_26 = NumpyArray::new();
    cpu_dwdh_anchors_26.from_file(&mut File::open("data/anchors_0").expect("anchors_0")).expect("Could not get anchors from file.");


    let mut anchors_ttensor_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Identity);

    cuda_error!(cudaMemcpy(anchors_ttensor_26.data, cpu_dwdh_anchors_26.data.as_ptr() as _, 4*3*2,
                          cudaMemcpyKind::cudaMemcpyHostToDevice));

    //TODO 09/17/21 I have yet to figure out how one broadcast tensors to a higher dimension
    //We are doing this instead.
    for i in 1..nxn*nxn {
        cuda_error!(cudaMemcpy(anchors_ttensor_26.data.offset((i*3*2*4) as _), anchors_ttensor_26.data, 4*3*2,
                              cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }
    

    let dims    = [nxn,nxn,3,1];
    let strides = [3*nxn*_nc,3*_nc, _nc, 1];
    let from_conf_desc26 = construct_ndtensor_descriptors(&dims, &org_strides26);
    let conf_desc26 = construct_ndtensor_descriptors(&dims, &strides);



    let dims = [nxn,nxn,3,_nc];
    let strides = [_nc*nxn*3, _nc*3,_nc,1];
    let mut conf_ttensor_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Sigmoid);
    let mut prob_ttensor_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Sigmoid);

    let from_prob_desc26 = construct_ndtensor_descriptors(&dims, &org_strides26);
    let prob_desc26 = construct_ndtensor_descriptors(&dims, &strides);


    let mut prob_ttensor_out_26 = GpuTensorTensor::construct(&cutensor_handle, &dims, GpuTensorOps::Sigmoid);


    let mut cpu_xygrid_26 = NumpyArray::new();
    cpu_xygrid_26.from_file(&mut File::open("data/xygrid_26").expect("xygrid_26")).expect("Could not get xy grids from file.");

    let mut xygrid_26_data : *mut std::ffi::c_void = null_mut(); 
    cuda_error!(cudaMalloc(&mut xygrid_26_data, (4*nxn*nxn*3*2) as _));
    cuda_error!(cudaMemcpy(xygrid_26_data,  cpu_xygrid_26.data.as_ptr() as _, (nxn*nxn*3*2*4) as _, 
                           cudaMemcpyKind::cudaMemcpyHostToDevice));




    //CONCAT prob and xywh
    let mut xy_final = NumpyArray::new();
    xy_final.dims.extend_from_slice(&[1, (26*26+13*13)*3, 2]);
    xy_final.data = vec![0u8; 4*(26*26+13*13)*3*2];

    let mut wh_final = NumpyArray::new();
    wh_final.dims.extend_from_slice(&[1, (26*26+13*13)*3, 2]);
    wh_final.data = vec![0u8; 4*(26*26+13*13)*3*2];

    let mut prob_final = NumpyArray::new();
    prob_final.dims.extend_from_slice(&[1, (26*26+13*13)*3, number_classes as _]);
    prob_final.data = vec![0u8; 4*(26*26+13*13)*3*number_classes];



    //NOTE
    //calc results

    {//Calc back-bone
        b1.calc(dnn_handle, workspace, workspace_size, &input_tensor);
        b2.calc(dnn_handle, workspace, workspace_size, &b1.output);
        b3.calc(dnn_handle, workspace, workspace_size, &b2.output);

        csp1.calc(dnn_handle, workspace, workspace_size, &b3.output);
        maxpooling1.calc(dnn_handle, &csp1.output, &mut maxpooling1_output);

        b4.calc(dnn_handle, workspace, workspace_size, &maxpooling1_output);

        csp2.calc(dnn_handle, workspace, workspace_size, &b4.output);
        maxpooling1.calc(dnn_handle, &csp2.output, &mut maxpooling2_output);

        b5.calc(dnn_handle, workspace, workspace_size, &maxpooling2_output);

        csp3.calc(dnn_handle, workspace, workspace_size, &b5.output);
        maxpooling1.calc(dnn_handle, &csp3.output, &mut maxpooling3_output);
       
        b6.calc(dnn_handle, workspace, workspace_size, &maxpooling3_output);
    }

    {//Calc lower branch
        l_b1.calc(dnn_handle, workspace, workspace_size, &b6.output);
        l_b2.calc(dnn_handle, workspace, workspace_size, &l_b1.output);
        bbox.calc(dnn_handle, workspace, workspace_size, &l_b2.output, &mut bbox_output);
    }

    {//Calc upper branch
        cuda_error!(cudaDeviceSynchronize());
        u_b1.calc(dnn_handle, workspace, workspace_size, &l_b1.output);
        cuda_error!(cudaDeviceSynchronize());


        let error = unsafe{ stbir_resize_float( u_b1.output.return_data().as_ptr(), 13, 13, 
                            4*13*128, resized.as_mut_ptr(),
                            26, 26, 4*26*128,
                            128) };

        if error == 0 {
            panic!("Could not resize!");
        }
        resized_gpu.set_data(&resized);

        concat_tensor(dnn_handle, &resized_gpu, &csp3.c3.output, &mut u_concat_tensor, u_inter_concat_descriptor1, u_inter_concat_descriptor2);

        u_b2.calc(dnn_handle, workspace, workspace_size, &u_concat_tensor);
        mbbox.calc(dnn_handle, workspace, workspace_size, &u_b2.output, &mut mbbox_output);
    }

    {
        ////////
        //13x13

        //split dxdy
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_dxdy_desc13, 
          bbox_output.data,
          (&0f32 as *const _) as *const _,
          dxdy_desc13,
          dxdy_ttensor_13.data,
        ));

        //split dwdh
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_dxdy_desc13, 
          bbox_output.data.offset(2*4),
          (&0f32 as *const _) as *const _,
          dxdy_desc13,
          dwdh_ttensor_13.data,
        ));

        //split conf
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_conf_desc13, 
          bbox_output.data.offset(4*4),
          (&0f32 as *const _) as *const _,
          conf_desc13,
          conf_ttensor_13.data,
        ));

        {//Broadcast 
            for i in 1..80 {
                dnn_error!(cudnnTransformTensor(
                  dnn_handle,
                  (&1f32 as *const _) as *const _,
                  from_conf_desc13, 
                  bbox_output.data.offset(4*4),
                  (&0f32 as *const _) as *const _,
                  conf_desc13,
                  conf_ttensor_13.data.offset(i*4),
                ));
            }
        }

        //split prob
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_prob_desc13, 
          bbox_output.data.offset(5*4),
          (&0f32 as *const _) as *const _,
          prob_desc13,
          prob_ttensor_13.data,
        ));

        ///////////
        //26x26

        //split dxdy
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_dxdy_desc26, 
          mbbox_output.data,
          (&0f32 as *const _) as *const _,
          dxdy_desc26,
          dxdy_ttensor_26.data,
        ));

        //split dwdh
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_dxdy_desc26, 
          mbbox_output.data.offset(2*4),
          (&0f32 as *const _) as *const _,
          dxdy_desc26,
          dwdh_ttensor_26.data,
        ));

        //split conf
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_conf_desc26, 
          mbbox_output.data.offset(4*4),
          (&0f32 as *const _) as *const _,
          conf_desc26,
          conf_ttensor_26.data,
        ));

        //Broadcasting
        for i in 1..80 {
            dnn_error!(cudnnTransformTensor(
              dnn_handle,
              (&1f32 as *const _) as *const _,
              from_conf_desc26, 
              mbbox_output.data.offset(4*4),
              (&0f32 as *const _) as *const _,
              conf_desc26,
              conf_ttensor_26.data.offset(i*4),
            ));
        }

        //split prob
        dnn_error!(cudnnTransformTensor(
          dnn_handle,
          (&1f32 as *const _) as *const _,
          from_prob_desc26, 
          mbbox_output.data.offset(5*4),
          (&0f32 as *const _) as *const _,
          prob_desc26,
          prob_ttensor_26.data,
        ));

        {
          //pred_xy = STRIDES[i] * ( tf.sigmoid(conv_raw_dxdy) * XYSCALE[i] - 0.5 * 
          //(XYSCALE[i] - 1) + xy_grid)

            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, &dxdy_ttensor_13, 
                                    &mut dxdy_ttensor_ones_13, Some(&mut dxdy_ttensor_out_13), GpuTensorOps::Add,
                                    1.05, 1f32);
            //panic!();


            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, 
                                    &GpuTensorTensor{ descriptor: dxdy_ttensor_out_13.descriptor, data: xygrid_13_data, .. GpuTensorTensor::default()}, //TODO should set this up above
                                    &mut dxdy_ttensor_out_13, 
                                    None, GpuTensorOps::Add,
                                    32f32, 32f32);


            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, 
                                    &dwdh_ttensor_13, //TODO should set this up above
                                    &mut anchors_ttensor_13, 
                                    Some(&mut GpuTensorTensor{ data: dwdh_ttensor_13.data, .. GpuTensorTensor::default()}), 
                                    GpuTensorOps::Mul, 1f32, 1f32);


        }

        {
         //pred_conf = tf.sigmoid(conv_raw_conf)
         //pred_prob = tf.sigmoid(conv_raw_prob)
         //pred_prob = pred_conf * pred_prob
            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, 
                                    &prob_ttensor_13, &mut conf_ttensor_13, Some(&mut prob_ttensor_out_13),
                                    GpuTensorOps::Mul, 1f32, 1f32);

        }
        {
          //pred_xy = STRIDES[i] * ( tf.sigmoid(conv_raw_dxdy) * XYSCALE[i] - 0.5 * 
          //(XYSCALE[i] - 1) + xy_grid)

            calc_elementwise_binary(&cutensor_handle, cuda_stream, 
                                    &dim_labels, &dxdy_ttensor_26, 
                                    &mut dxdy_ttensor_ones_26, 
                                    Some(&mut dxdy_ttensor_out_26), 
                                    GpuTensorOps::Add, 1.05, 1f32);



            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, 
                                    &GpuTensorTensor{ descriptor: dxdy_ttensor_out_26.descriptor, data: xygrid_26_data, .. GpuTensorTensor::default()}, //TODO should set this up above
                                    &mut dxdy_ttensor_out_26, 
                                    None, GpuTensorOps::Add,
                                    32f32, 32f32);

            //pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, 
                                    &dwdh_ttensor_26, //TODO should set this up above
                                    &mut anchors_ttensor_26, 
                                    Some(&mut GpuTensorTensor{ data: dwdh_ttensor_26.data, .. GpuTensorTensor::default()}), 
                                    GpuTensorOps::Mul, 1f32, 1f32);


        }
        {
         //pred_conf = tf.sigmoid(conv_raw_conf)
         //pred_prob = tf.sigmoid(conv_raw_prob)
         //pred_prob = pred_conf * pred_prob
            calc_elementwise_binary(&cutensor_handle, cuda_stream, &dim_labels, 
                                    &prob_ttensor_26, &mut conf_ttensor_26, Some(&mut prob_ttensor_out_26),
                                    GpuTensorOps::Mul, 1f32, 1f32);

        }
    }
    cuda_error!(cudaDeviceSynchronize());


    cuda_error!(cudaMemcpy(prob_final.data.as_ptr() as _, prob_ttensor_out_26.data, 4*26*26*3*80, 
                           cudaMemcpyKind::cudaMemcpyDeviceToHost));
    cuda_error!(cudaMemcpy(prob_final.data.as_ptr().offset(4*26*26*3*80) as _, prob_ttensor_out_13.data, 4*13*13*3*80, 
                           cudaMemcpyKind::cudaMemcpyDeviceToHost));


    cuda_error!(cudaMemcpy(xy_final.data.as_ptr() as _, dxdy_ttensor_out_26.data, 4*26*26*3*2, 
                           cudaMemcpyKind::cudaMemcpyDeviceToHost));
    cuda_error!(cudaMemcpy(xy_final.data.as_ptr().offset(4*26*26*3*2) as _, dxdy_ttensor_out_13.data, 4*13*13*3*2, 
                           cudaMemcpyKind::cudaMemcpyDeviceToHost));


    cuda_error!(cudaMemcpy(wh_final.data.as_ptr() as _, dwdh_ttensor_26.data, 4*26*26*3*2, 
                           cudaMemcpyKind::cudaMemcpyDeviceToHost));
    cuda_error!(cudaMemcpy(wh_final.data.as_ptr().offset(4*26*26*3*2) as _, dwdh_ttensor_13.data, 4*13*13*3*2, 
                               cudaMemcpyKind::cudaMemcpyDeviceToHost));



    //TODO do better mask
    
    let mut scores_and_bboxes : Vec::<ScoreBoundingBox> = vec![];
    //TODO remove mask and just iterate over indices
    let mut mask = vec![false; (26*26+13*13)*3];
    for i in 0..mask.len(){

        for j in 0..number_classes{
            let a = prob_final.get(i*80+j);

            if a > 0.4 {
                mask[i] = true;

                let rect = {
                    let w = wh_final.get(i*2+j)/416f32;
                    let h = wh_final.get(i*2+j+1)/416f32;

                    let x = xy_final.get(i*2+j)/416f32 - w/2f32; 
                    let y = xy_final.get(i*2+j+1)/416f32 - h/2f32;  

                    [x,y,w,h]
                };

                scores_and_bboxes.push(ScoreBoundingBox{ score: a, bounding_box: rect, class: j });
            }
        }
    }

    //NOTE
    //draw new image.
    change_font(&FONT_NOTOSANS).expect("");

    let classes = ["person","bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple","sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    let scores_and_bboxes = soft_nms(scores_and_bboxes, 0.5);

    println!("{:?}", &scores_and_bboxes);
    //NOTE draw bounding box
    for it in scores_and_bboxes.iter(){
        if it.score < 0.6 { continue; }
        let [xf32, yf32, wf32, hf32] = it.bounding_box;

        let x = (xf32.max(0f32).min(1f32) * org_img_width as f32) as usize;
        let w = (wf32.max(0f32).min(1f32) * org_img_width as f32) as usize;
        let y = (yf32.max(0f32).min(1f32) * org_img_height as f32) as usize;
        let h = (hf32.max(0f32).min(1f32) * org_img_height as f32) as usize;

        let x0 = x;
        let x1 = (x + w).min(org_img_width as usize);
        let y0 = y;
        let y1 = (y + h).min(org_img_height as usize);

        let color = [250, 0, 0];

        for i in y0..y1 {
            for j in x0..x1 {
                if (i > y0+2 && i < y1-2) 
                && (j > x0+2 && j < x1-2){
                    continue;
                }
                org_img.buffer[ 3*(i*(org_img_width as usize) + j) + 0] = color[0];
                org_img.buffer[ 3*(i*(org_img_width as usize) + j) + 1] = color[1];
                org_img.buffer[ 3*(i*(org_img_width as usize) + j) + 2] = color[2];

                //put color
            }
        }
        draw_string( &mut org_img, classes[it.class], x0 as i32, y0 as i32, [1f32, 0f32, 0f32, 1f32], 24f32);
        draw_string( &mut org_img, &format!("{:.2}", it.score), x0 as i32 + 12, y0 as i32+12, [0f32, 0f32, 1f32, 1f32], 18f32);
    }

    write_png(output_file, org_img_width, org_img_height, 3, &org_img.buffer, 3*org_img_width); 

    print_alloc();
    b1.drop();
    b2.drop();
    b3.drop();

    csp1.drop();
    b4.drop();

    csp2.drop();
    b5.drop();

    csp3.drop();
    b6.drop();

    l_b1.drop();
    l_b2.drop();
    bbox.drop();
    bbox_output.drop();

    u_b1.drop();
    u_b2.drop();
    mbbox.drop();
    mbbox_output.drop();

    dxdy_ttensor_13.drop();
    dxdy_ttensor_ones_13.drop();
    dxdy_ttensor_out_13.drop();
    anchors_ttensor_13.drop();

    dxdy_ttensor_26.drop();
    dxdy_ttensor_ones_26.drop();
    dxdy_ttensor_out_26.drop();
    anchors_ttensor_26.drop();


    cuda_error!(cudaFree(workspace));
    cuda_error!(cudaStreamDestroy(cuda_stream));
    dnn_error!(cudnnDestroy(dnn_handle));

}



pub fn overlap_rect_area(rect1: [f32;4], rect2: [f32;4])->f32{
    let ol_x = f32::max(0f32, f32::min(rect1[0]+rect1[2], rect2[0]+rect2[2]) - f32::max(rect1[0], rect2[0]));
    let ol_y = f32::max(0f32, f32::min(rect1[1]+rect1[3], rect2[1]+rect2[3]) - f32::max(rect1[1], rect2[1]));

    return ol_x * ol_y;
}
pub fn union_rect_area(rect1: [f32;4], rect2: [f32;4])->f32{
    let a1 = rect1[2] * rect1[3];
    let a2 = rect2[2] * rect2[3];

    let ov = overlap_rect_area(rect1, rect2);
    if ov <= 0f32 {
        return 0f32;
    }

    let union_area = a1+a2-ov;
    return union_area;
}

pub fn soft_nms(mut input: Vec<ScoreBoundingBox>, iou_score: f32 )->Vec<ScoreBoundingBox>{
     let mut rt = Vec::new();
     while input.len() > 0 {
        //
        let mut max_s = 0f32;
        let mut max_s_index = 9999;

        for (i, it) in input.iter().enumerate(){
            if it.score >= max_s {
                max_s_index = i;
                max_s = it.score;
            }
        }

        let b = input.remove(max_s_index);
        for it in input.iter_mut(){
            let intersection = overlap_rect_area(b.bounding_box, it.bounding_box);
            if intersection <= 0f32 {
                continue;
            }
            let union = union_rect_area(b.bounding_box, it.bounding_box);
            let iou = intersection/union;

            if iou > iou_score {
                it.score = (1f32 - iou) * it.score;
            }
        } 
        rt.push(b);
     }

     rt
}

#[derive(Debug)]
pub struct ScoreBoundingBox{
    pub score: f32,
    pub bounding_box: [f32;4],
    pub class: usize,
}

pub struct Image{
    w: i32,
    h: i32,
    buffer: Vec::<u8>,
}

//TODO
static mut GLOBAL_FONTINFO    : stbtt_fontinfo  = new_stbtt_fontinfo();
static mut FONT_BUFFER        : Vec<u8> = Vec::new();
const FONT_NOTOSANS : &[u8] = std::include_bytes!("../assets/NotoSans-Regular.ttf");//TODO better pathing maybe

pub fn change_font(buffer: &[u8])->Result<(), &str>{unsafe{


    FONT_BUFFER.clear(); 
    FONT_BUFFER.extend_from_slice(buffer);
   
    if stbtt_InitFont(&mut GLOBAL_FONTINFO as *mut stbtt_fontinfo, FONT_BUFFER.as_ptr(), 0) == 0{
        println!("font was not able to load.");
        return Err("Font was not able to be loaded.");
    }
    return Ok(());
}}

/// Draws the provided character to the canvas. `size` is rounded to the nearest integer. 
/// Returns character width in pixels.
pub fn draw_char( canvas: &mut Image, character: char, mut x: i32, mut y: i32,
             color: [f32; 4], mut size: f32 )->i32{unsafe{

    //NOTE Check that globalfontinfo has been set
    if GLOBAL_FONTINFO.data == null_mut() {
        println!("Global font has not been set.");
        return -1;
    }


    let dpmm_ratio = 1f32;

    size = size.round();
    let dpmm_size = (dpmm_ratio * size).round();

    x = (dpmm_ratio * x as f32).round() as _;
    y = (dpmm_ratio * y as f32).round() as _;


    //Construct a char buffer
    let char_buffer;
    let cwidth;
    let cheight;
    let scale;
    {
        let mut x0 = 0i32;
        let mut x1 = 0i32;
        let mut y0 = 0i32;
        let mut y1 = 0i32;
        let mut ascent = 0;
        let mut descent = 0;


        stbtt_GetFontVMetrics(&mut GLOBAL_FONTINFO as *mut stbtt_fontinfo,
                              &mut ascent as *mut i32,
                              &mut descent as *mut i32, null_mut());
        scale = stbtt_ScaleForPixelHeight(&GLOBAL_FONTINFO as *const stbtt_fontinfo, dpmm_size);
        let baseline = (ascent as f32 * scale ) as i32;

        cwidth = (scale * (ascent - descent) as f32 ) as usize + 4; //NOTE buffer term should be reduced.
        cheight = (scale * (ascent - descent) as f32 ) as usize + 4;//NOTE buffer term should be reduced.


        let glyph_index = stbtt_FindGlyphIndex(&GLOBAL_FONTINFO as *const stbtt_fontinfo, character as i32);
        
        //TODO remove this, it is not nessary for this application
        char_buffer = {
            let mut _char_buffer = vec![0u8; cwidth * cheight];
            stbtt_GetGlyphBitmapBoxSubpixel(&GLOBAL_FONTINFO as *const stbtt_fontinfo, glyph_index, scale, scale, 0.0,0.0,
                                                    &mut x0 as *mut i32,
                                                    &mut y0 as *mut i32,
                                                    &mut x1 as *mut i32,
                                                    &mut y1 as *mut i32);
            stbtt_MakeGlyphBitmapSubpixel( &GLOBAL_FONTINFO as *const stbtt_fontinfo,
                                           &mut _char_buffer[cwidth*(baseline + y0) as usize + (5 + x0) as usize ] as *mut u8,
                                           x1-x0+2, y1-y0, cwidth as i32, scale, scale,0.0, 0.0, glyph_index);
            _char_buffer
        };

    }

    //NOTE
    //The character will not render if invisible.
    
    if character as u8 > 0x20{   //render char_buffer to main_buffer
        let buffer = &mut canvas.buffer;
        let gwidth = canvas.w as isize;
        let gheight = canvas.h as isize;
        let offset = (x as isize + y as isize * gwidth);


        let a = color[3];
        let orig_r = color[0] * a;
        let orig_g = color[1] * a;
        let orig_b = color[2] * a;




        let y_is = y as isize;
        let x_is = x as isize;
        for i in 0..cheight as isize{
            if i + y_is > gheight {continue;}
            if i + y_is <= 0 {continue;}

            for j in 0..cwidth as isize{

                if (j + i*gwidth + offset) > gwidth * gheight {continue;}

                if j + x_is  > gwidth {continue;}
                if j + x_is  <= 0 {continue;}

                let mut text_alpha = char_buffer[j as usize + cwidth * (i as usize )] as f32;
                //let mut text_alpha = char_buffer[j as usize + cwidth * (cheight - 1 - i as usize)] as f32;
                let r = (orig_r * text_alpha) as u8;
                let g = (orig_g * text_alpha) as u8;
                let b = (orig_b * text_alpha) as u8;

                if 3*(j + i*gwidth + offset) as usize + 3 > buffer.len() {
                    continue;
                }
                let dst_rgb = &mut buffer[3*(j + i*gwidth + offset) as usize.. 3*(j + i*gwidth + offset) as usize + 3];

                text_alpha = (255.0 - text_alpha * a) / 255.0;
                
                dst_rgb[0] = r + ( dst_rgb[0] as f32 * text_alpha ) as u8;
                dst_rgb[1] = g + ( dst_rgb[1] as f32 * text_alpha ) as u8;
                dst_rgb[2] = b + ( dst_rgb[2] as f32 * text_alpha ) as u8;


            }
        }
    }

    let mut adv : i32 = 0;
    let mut lft_br : i32 = 0; // NOTE: Maybe remove this
    stbtt_GetCodepointHMetrics(&GLOBAL_FONTINFO as *const stbtt_fontinfo, character as i32, &mut adv as *mut i32, &mut lft_br as *mut i32);
    return (adv as f32 * scale) as i32;
}}


/// Draws the string to the canvas provided. Returns string width in pixels.
/// Position values x and y are indicate where the string will begin.
/// NOTE there is about a 4 pixel buffer between x and the first pixel the function is able to draw
/// to.
pub fn draw_string( canvas: &mut Image, string: &str, x: i32, y: i32,
             color: [f32; 4], size: f32 )->i32{
    let mut offset = 0;
    for it in string.chars(){
        offset += draw_char(canvas, it, x + offset, y, color, size);
    }
    return offset;
}


