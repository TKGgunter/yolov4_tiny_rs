#![allow(non_snake_case, non_camel_case_types, dead_code)]


#[repr(C)]
pub struct stbtt__buf{
   data: *mut u8,
   cursor: i32,
   size: i32,
}

#[inline]
pub const fn new_stbtt__buf()->stbtt__buf{
     stbtt__buf{
       data: std::ptr::null_mut(),
       cursor: 0,
       size: 0,
    }
}

#[repr(C)]
pub struct stbtt_fontinfo{
   userdata: *mut std::ffi::c_void,
   pub data: *mut u8,              // pointer to .ttf file
   fontstart: i32,        // offset of start of font

   umGlyphs: i32,                    // number of glyphs, needed for range checking

   loca: i32,
   head: i32,
   glyf: i32,
   hhea: i32,
   hmtx: i32,
   kern: i32,
   gpos: i32, // table locations as offset from start of .ttf
   index_map: i32,                     // a cmap mapping for our chosen character encoding
   indexToLocFormat: i32,              // format needed to map from glyph index to glyph

   cff:         stbtt__buf,                    // cff font data
   charstrings: stbtt__buf,            // the charstring index
   gsubrs:      stbtt__buf,                 // global charstring subroutines index
   subrs:       stbtt__buf,                  // private charstring subroutines index
   fontdicts:   stbtt__buf,              // array of font dicts
   fdselect:    stbtt__buf,               // map from glyph to fontdict
}

#[inline]
pub const fn new_stbtt_fontinfo()->stbtt_fontinfo{
    stbtt_fontinfo{
       userdata: std::ptr::null_mut(),
       data: std::ptr::null_mut(),              // pointer to .ttf file
       fontstart: 0,        // offset of start of font

       umGlyphs: 0,                    // number of glyphs, needed for range checking

       loca: 0,
       head: 0,
       glyf: 0,
       hhea: 0,
       hmtx: 0,
       kern: 0,
       gpos: 0, // table locations as offset from start of .ttf
       index_map: 0,                     // a cmap mapping for our chosen character encoding
       indexToLocFormat: 0,              // format needed to map from glyph index to glyph

       cff:         new_stbtt__buf(),                    // cff font data
       charstrings: new_stbtt__buf(),            // the charstring index
       gsubrs:      new_stbtt__buf(),                 // global charstring subroutines index
       subrs:       new_stbtt__buf(),                  // private charstring subroutines index
       fontdicts:   new_stbtt__buf(),              // array of font dicts
       fdselect:    new_stbtt__buf(),               // map from glyph to fontdict
    }
}


pub fn new__buf()->stbtt__buf{
    stbtt__buf{
       data: std::ptr::null_mut(),
       cursor: 0,
       size: 0,
    }

}
pub fn new_fontinfo()->stbtt_fontinfo{
    stbtt_fontinfo{
        userdata: std::ptr::null_mut(),
        data: std::ptr::null_mut(),
        fontstart: 0,
        umGlyphs: 0,
        loca: 0,
        head: 0,
        glyf: 0,
        hhea: 0,
        hmtx: 0,
        kern: 0,
        gpos: 0,
        index_map: 0,
        indexToLocFormat: 0,
        cff: new__buf(),                    // cff font data
        charstrings: new__buf(),            // the charstring index
        gsubrs: new__buf(),                 // global charstring subroutines index
        subrs: new__buf(),                  // private charstring subroutines index
        fontdicts: new__buf(),              // array of font dicts
        fdselect: new__buf(),               // map from glyph to fontdict

    }
}

extern{
    pub fn stbtt_InitFont(font: *mut stbtt_fontinfo, buffer: *const u8, offset: i32)->i32;

    pub fn stbtt_ScaleForPixelHeight(font : *const stbtt_fontinfo, size: f32)->f32;
    pub fn stbtt_GetFontVMetrics(font: *const stbtt_fontinfo, ascent: *mut i32, descent: *mut i32, lineGap: *mut i32);

    pub fn stbtt_GetCodepointHMetrics(font: *const stbtt_fontinfo, char: i32, advance: *mut i32, leftSideBearing: *mut i32);
    pub fn stbtt_GetCodepointBitmapBoxSubpixel(font: *const stbtt_fontinfo, char: i32, scale_x: f32, scale_y: f32, x_shift: f32, y_shift: f32,
          x0: *mut i32, y0: *mut i32, x1: *mut i32, y1: *mut i32);
    pub fn stbtt_MakeCodepointBitmapSubpixel(stbtt_fontinfo:  *const stbtt_fontinfo, output: *mut u8, out_w: i32, out_h: i32, out_stride: i32,
         scale_x: f32, scale_y: f32, x_shift: f32, y_shift: f32,codepoint: i32);
    pub fn stbtt_GetFontBoundingBox( info: *const stbtt_fontinfo, x0: *mut i32, y0: *mut i32, x1: *mut i32, y1: *mut i32);
    
    pub fn stbtt_FindGlyphIndex( info: *const stbtt_fontinfo, unicode_codepoint: i32)->i32;
    pub fn stbtt_GetGlyphHMetrics(info: *const stbtt_fontinfo, glyph_index: i32, advanceWidth: *mut i32, leftSideBearing: *mut i32);
    pub fn stbtt_GetGlyphBitmapBoxSubpixel(font: *const stbtt_fontinfo, glyph: i32, scale_x: f32, scale_y: f32, x_shift: f32, y_shift: f32,
          x0: *mut i32, y0: *mut i32, x1: *mut i32, y1: *mut i32);
    pub fn stbtt_MakeGlyphBitmapSubpixel(stbtt_fontinfo:  *const stbtt_fontinfo, output: *mut u8, out_w: i32, out_h: i32, out_stride: i32,
         scale_x: f32, scale_y: f32, x_shift: f32, y_shift: f32, glyph: i32);
}
