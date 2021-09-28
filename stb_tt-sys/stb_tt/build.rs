extern crate cc;

fn main() {
    println!("\n\n\n\nAre we running?\n\n\n\n");


    match pkg_config::find_library("stb_tt"){
        Ok(_) => return,
        Err(..)=> {}
    }
    cc::Build::new()
        .file("stb_tt/stb_truetype.c")
        //.define("_USRDLL", None)
        //.define("STB_TT_EXPORTS", None)
        //.include(&root.join("lib"))
        .opt_level(2)
        .compile("stb_tt");
}
