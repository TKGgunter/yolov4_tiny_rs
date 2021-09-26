
fn main() {
    
    #[cfg(target_os = "linux")]
    {
        //println!("cargo:include={}", "/usr/local/cuda/include");
        println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");
    }
   
    
    #[cfg(target_os = "windows")]
    { 
        println!("cargo:rustc-link-search=native={}", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.4\\lib\\x64\\");
    }
    println!("cargo:rustc-link-lib=dylib={}", "cuda");
    println!("cargo:rerun-if-changed=build.rs");
    
}
