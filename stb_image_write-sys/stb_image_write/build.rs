extern crate cc;

fn main() {
    println!("\n\n\n\nAre you winning?\n\n\n\n");


    match pkg_config::find_library("stb_image_write"){
        Ok(_) => return,
        Err(..)=> {}
    }
    cc::Build::new()
        .file("stb_image_write/stb_image_write.c")
        .opt_level(2)
        .compile("stb_image_write");
}
