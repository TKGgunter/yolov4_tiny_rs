#![allow(non_snake_case, non_camel_case_types, dead_code)]


extern {
    pub fn stbir_resize_float( input_pixels: *const f32, input_w: i32, input_h: i32, 
                               input_stride_in_bytes: i32, output_pixels: *mut f32, 
                               output_w: i32, output_h: i32, output_stride_in_bytes: i32,
                               num_channels: i32)->i32;
}


#[test]
fn it_works() {unsafe{
    
    use std::time::Instant;

    let arr = [1f32; 13*13*128];
    let mut rt = [0f32; 4*13*13*128];

    let mut inst = Instant::now();
    let error = stbir_resize_float( arr.as_ptr(), 13, 13, 4*13*128, 
                                    rt.as_mut_ptr(), 26, 26, 4*13*128,
                                    128);

    println!("{} {:?} {:?}", error, &rt[..5], inst.elapsed());
    panic!();
}}
