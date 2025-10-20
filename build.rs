fn main() {
    cc::Build::new()
        .file("c_code/f_hhh.c")
        .file("c_code/nrutil.c")
        .flag("-Wno-unused-variable")
        .flag("-Wno-unused-parameter")
        .compile("f_hhh");
}
