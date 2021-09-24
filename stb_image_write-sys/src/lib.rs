#![allow(non_snake_case, non_camel_case_types, dead_code)]

extern{
     pub fn stbi_write_png(filename: *const i8, w: i32, h: i32, comp: i32, data: *const std::ffi::c_void, stride_in_bytes: i32)->i32;
     pub fn stbi_write_bmp(filename: *const i8, w: i32, h: i32, comp: i32, data: *const std::ffi::c_void)->i32;
     pub fn stbi_write_jpg(filename: *const i8, w: i32, h: i32, comp: i32, data: *const std::ffi::c_void, quality: i32)->i32;
     pub fn stbi_flip_vertically_on_write(flag: i32); // flag is non-zero to flip data vertically
}

pub fn write_png(filename: &str, w: i32, h: i32, channels: i32, data: &[u8], stride_in_bytes: i32){unsafe{
     let error = stbi_write_png(std::ffi::CString::new(filename).unwrap().as_ptr() as _, w, h, 
                                channels, (data.as_ptr() as *const u8) as _, stride_in_bytes);

     //TODO handle error properly
     if error == 0 {
        panic!("there was some problem writing file.");
     }
}}



