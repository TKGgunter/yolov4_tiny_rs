
use std::fs::File;
use std::io::prelude::*;
use std::mem::size_of;
use std::mem::transmute;



pub struct NumpyArray{
    pub ndims: u32,
    pub dims : Vec<u32>,
    pub _type: u32, //TODO make this an enum
    pub data : Vec<u8>,
}


impl NumpyArray{
    pub fn new()->NumpyArray{
        NumpyArray{
            ndims: 0u32,
            dims : vec![],
            _type: 0u32,
            data : vec![],
        }
    }
    //TODO should be a mut ref and use the indexing impl to make things easy for user.
    pub fn get(&self, i: usize)->f32{unsafe{
        let l = self.data.len();
        let a = transmute::<&[u8], &[f32]>(&self.data);
        a[i]
        
    }}

    pub fn print_dataf32(&self, range: Option<(usize, usize)>){unsafe{
        let l = self.data.len();
        let a = transmute::<&[u8], &[f32]>(&self.data);

        let mut begin = 0;
        let mut end = l/4;

        match range {
            Some(o)=> {
                if o.0 > begin {
                    begin = o.0;
                }
                if o.1 < end {
                    end = o.1;
                }
            },
            _=>{}
        }

        print!("NumpyArray[");
        for i in begin..end{
            print!("{:?}, ", a[i]);
        }
        print!("] len: {:?}", self.data.len());

        println!("");
    }}
    

    pub fn size_btyes(&self)->usize{
        self.data.len()
    }

    pub fn from_file(&mut self, f: &mut File)->Result<(), ()>{unsafe{
        //Load ndims from file to array
        macro_rules! _send_break_command {
            ($x:expr)=>{
                match $x {
                    Ok(_)=>{},
                    _=>{return Err(());}
                }
            }
        }

        let mut _ndims = transmute::<_, &mut [u8; 4]>(&mut self.ndims);
        _send_break_command!( f.read_exact(_ndims) );

        //load dims to dyarray
        {
            self.dims = vec![0; self.ndims as _];
            let mut _dims = std::slice::from_raw_parts_mut( self.dims.as_mut_ptr() as *mut u8 , (4*self.ndims) as _ );
            _send_break_command!( f.read_exact(&mut _dims) );
        }

        //load type
        let mut _type = transmute::<_, &mut [u8; 4]>(&mut self._type);
        _send_break_command!( f.read_exact(_type) );

        //data
        let bytes_to_load = {
            let mut a = 1;
            for it in self.dims.iter(){
                a *= *it;
            }
            a *= 4; //TODO  handle different types
            a
        };
        self.data = vec![0u8; bytes_to_load as usize];
        _send_break_command!( f.read_exact(&mut self.data) );

        
        if false {
        //TODO this is for testing purposes
        unsafe{
            println!("array values: ");
            let bytes_to_load = {
                let mut a = 1;
                for jt in self.dims.iter(){
                    a *= *jt;
                }
                a *= 4; //TODO  handle different types
                a
            };
            
            let mut _data = std::slice::from_raw_parts( self.data.as_ptr() as *const f32, bytes_to_load as _ );
            println!("{:?}", (self.ndims, &self.dims, self._type, &_data[..5]));
        }
        }
        return Ok(());
    }}
    pub fn load_data(file_name: &str)->Vec<Self>{
        //LOAD FILE handle
        let mut f = File::open(file_name).expect("Could not open file.");


        let mut rt = vec![];
        loop/*still more bytes to read*/{unsafe{
            let mut array = NumpyArray::new();

            match array.from_file(&mut f){
                Result::Err(_)=>break,
                _=>(),
            }
            rt.push(array);
        }}

        if false {
        //TODO this is for testing purposes
        for it in rt.iter(){unsafe{
            let bytes_to_load = {
                let mut a = 1;
                for jt in it.dims.iter(){
                    a *= *jt;
                }
                a *= 4; //TODO  handle different types
                a
            };
            
            let mut _data = std::slice::from_raw_parts( it.data.as_ptr() as *const f32, bytes_to_load as _ );
            println!("{:?}", (it.ndims, &it.dims, it._type, &_data[..5]));
        }}
        }
        
        return rt;
    }

}

