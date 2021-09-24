#![allow(non_snake_case, non_camel_case_types, dead_code)]

extern{
    pub fn stbi_load_from_memory( buffer: *const u8, len: i32, 
                                  w: *mut i32, h: *mut i32, 
                                  channels_in_file: *mut i32, 
                                  desired_channels: i32)->*mut u8;
    pub fn stbi_failure_reason()->*const i8;
    pub fn stbi_is_hdr_from_memory(buffer: *const u8, len: i32)->i32;
    pub fn stbi_set_flip_vertically_on_load(flag_true_if_should_flip: i32);
    pub fn stbi_image_free(retval_from_stbi_load : *const std::ffi::c_void);
}


#[derive(Clone)]
pub struct StbiImage{
    pub width: i32,
    pub height: i32,
    pub bits_per_pixel: i32,
    pub buffer: Vec<u8>,
}

pub fn stbi_load_from_memory_32bit(buffer: &[u8])->Result<StbiImage, String>{unsafe{
    let mut image = StbiImage{width: 0, height: 0, bits_per_pixel: 0, buffer: Vec::new() };
    
    let len = buffer.len() as i32;
    let mut channels_in_file = 0;
    let image_buffer = stbi_load_from_memory( buffer.as_ptr(), len, 
                                              &mut image.width as *mut _, 
                                              &mut image.height as *mut _, 
                                              &mut channels_in_file as *mut i32, 
                                              4);

    image.bits_per_pixel = 32;


    if image_buffer == std::ptr::null_mut(){
        let cstr =  std::ffi::CStr::from_ptr(crate::stbi_failure_reason()); 
        return Err(cstr.to_str().unwrap().to_owned());//TODO
    }

    image.buffer = vec![0; (image.width*image.height*4) as usize];
    std::ptr::copy_nonoverlapping( image_buffer, image.buffer.as_mut_ptr(), (image.width*image.height*4) as usize);
    

    stbi_image_free(image_buffer as *mut _);
    return Ok(image); 
}}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {unsafe{
        use std::fs::File;
        use std::io::prelude::*;
        use std::ffi::CStr;

        let mut w = 0i32;
        let mut h = 0i32;
        let mut channels_in_file :i32= 0;

        let mut f = File::open("../assets/resistor.bmp").expect("BMP file could not be opened.");
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer).expect("Buffer could not be read.");

        let image = crate::stbi_load_from_memory_32bit(&buffer);

        if image.is_err(){ 
            assert!(false);
        }
    }}
}
