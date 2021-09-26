
extern crate cc;

fn main() {
    println!("\n\n\n\nAre you winning?\n\n\n\n");


    match pkg_config::find_library("stb_resize"){
        Ok(_) => return,
        Err(..)=> {}
    }
    cc::Build::new()
        .file("stb_resize/stb_resize.c")
        .opt_level(2)
        .compile("stb_resize");
}
