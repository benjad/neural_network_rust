use std::io;
use  std::f64::consts;
use rand::Rng;
use ndarray::{arr1,arr2};
use ndarray::{Axis,Array, Array2,ArrayView1, ArrayView2, Slice};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;



fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    run_network();

}

fn l1_norm(x: ArrayView2<f64>) -> f64 {
    x.fold(0., |acc, elem| acc + elem.abs())
}

fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x).abs()
}

fn relu(x:&ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|elem|  elem.max(0.00))
}

fn softmax(x:&ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    //let axis_sum = x.fold_axis(0., Axis(1),|sum,elem| sum +  consts::E.powf(*elem));
    let max_v = &x.iter().fold(0.0/0.0, |m, v| v.max(m));
    let x_e = x.map(|elem|  consts::E.powf(*elem - max_v));
    let axis_sum = x.map_axis(Axis(0),|view| 1.0 / view.map( |i| consts::E.powf(*i - max_v)).sum() );
    let test = Array2::from_diag(&axis_sum);
    let result = x_e.dot(&test);
    return result;
}

fn foward_prop(x:&ndarray::Array2<f64>,w1:&ndarray::Array2<f64>,b1:&ndarray::Array2<f64>,w2:&ndarray::Array2<f64>,b2:&ndarray::Array2<f64>)->
             (ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>) { 
    let Z1 = w1.dot(x) + b1;
    let A1 = relu(&Z1);
    let Z2 = w2.dot(&A1) + b2;
    let A2 = softmax(&Z2);
    return (Z1,A1,Z2,A2);

}

fn back_prop(z1:&ndarray::Array2<f64>,a1:&ndarray::Array2<f64>,z2:&ndarray::Array2<f64>,a2:&ndarray::Array2<f64>,w2:&ndarray::Array2<f64>,x:&ndarray::Array2<f64>,y:&ndarray::Array1<i32>)->
(ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>) {
    let m = y.shape()[0] as f64;
    let y_hot = one_hot(y);
    let dZ2 = a2 - y_hot;
    let dW2 = 1.0 / m * (dZ2.dot(&a1.t()));
    //let db2 = 1.0 / m * dZ2.fold_axis(Axis(1),0.0, |a , i| arr1(&[a]) + i); // convertir este array de una fila largo n en una matrix 1xn
    let mut db2  = Array::from_shape_vec((10, 1), (dZ2.fold_axis(Axis(1),0.0, |a , i| a + i)).to_vec()).unwrap();
    let db2 =  1.0 / m * db2; 
    let dZ1 = w2.t().dot(&dZ2) * (relu_dev(z1));
    let dW1 = 1.0 / m * dZ1.dot(&x.t());
    let mut db1  = Array::from_shape_vec((10, 1), (dZ1.fold_axis(Axis(1),0.0, |a , i| a + i)).to_vec()).unwrap();

    let db1 =  1.0 / m * &db2;
    return (dW1, db1, dW2, db2);
}

fn update_params(w1:&ndarray::Array2<f64>,b1:&ndarray::Array2<f64>,w2:&ndarray::Array2<f64>,b2:&ndarray::Array2<f64>,
                 dw1:&ndarray::Array2<f64>,db1:&ndarray::Array2<f64>,dw2:&ndarray::Array2<f64>,db2:&ndarray::Array2<f64>,alpha:f64) ->
                 (ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>) {
    let mut w1_m = w1 - alpha * dw1;
    let mut b1_m = b1 - alpha * db1;
    let mut w2_m = w2 - alpha * dw2;
    let mut b2_m = b2 - alpha * db2;
    return (w1_m, b1_m, w2_m, b2_m);            
}


fn relu_dev(x:&ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let result = x.map(|&i| {
        if i > 0.0 {
            1.0
        } else {
          0.0
        }
    });
    return result;  
}

fn predictions(x: ndarray::Array2<f64>) -> ndarray::Array1<i32> {
    let lenght = x.shape()[1];
    let mut results = Array::<i32,_>::zeros(lenght);
    for (i, col) in x.axis_iter(Axis(1)).enumerate() {
        let (max_idx, max_val) =
            col.iter()
                .enumerate()
                .fold((0, col[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        results[i] = max_idx as i32;
    }
    return results;
}


fn gradient_descent(x:&ndarray::Array2<f64>,y:&ndarray::Array1<i32>, iter:i32, alpha:f64) ->
                    (ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>,ndarray::Array2<f64>){
    
    let cols = x.shape()[1];
    println!("numbers of rows: {}", &cols);   
    let mut z1 =  Array2::<f64>::zeros((10, cols));
    let mut a1 =  Array2::<f64>::zeros((10, cols));
    let mut z2 =  Array2::<f64>::zeros((10, cols));
    let mut a2 =  Array2::<f64>::zeros((10, cols));
    let mut foward_array = (z1,a1,z2,a2);
    
    let repeater = Array2::<f64>::ones((1, cols));
    let mut W1 = Array2::random((10,784), Uniform::new(-0.5, 0.5));
    let mut B1 = (Array2::random((10,1), Uniform::new(-0.5, 0.5))).dot(&repeater);
    let mut W2 = Array2::random((10,10), Uniform::new(-0.5, 0.5));
    let mut B2 = (Array2::random((10,1), Uniform::new(-0.5, 0.5))).dot(&repeater);

    for i in 0..iter {
        foward_array = foward_prop(&x,&W1,&B1,&W2,&B2);    
        // println!("matrix A2: {}", &W1);
        let (dw1,db1,dw2,db2) = back_prop(&foward_array.0, &foward_array.1, &foward_array.2, &foward_array.3, &W2, &x, &y);
        let params = update_params(&W1,&B1,&W2,&B2,&dw1,&db1,&dw2,&db2,alpha);
        W1 = params.0;
        B1 = params.1;
        W2 = params.2;
        B2 = params.3;
        println!("value of iterator is: {}", i);   

    }
    return (W1, B1, W2, B2)
}

fn one_hot(y:&ndarray::Array1<i32>) -> ndarray::Array2<f64>{
    // let len = x.shape()[0];
    let siz = y.iter().cloned().count();
    let max = itertools::max(y).unwrap();
    let max_usize: usize = *max as usize + 1;
    let mut  board = Array2::<f64>::zeros((max_usize, siz));
    let mut count: usize = 0 ;
    for elem in y.iter() {
        board[[*elem as usize, count]] = 1 as f64;
        count = count + 1;  
    }
    return board;
}


fn accuracy(y_pred:&ndarray::Array1<i32>, y_true:&ndarray::Array1<i32>) -> f64{
    let zero = y_pred - y_true;
    let len = zero.shape()[0] as f64;
    let n_zero =  zero.fold(0.0, |sum, elem|  {
        if elem == &0 {
            sum + 1.0
        } else {
            sum + 0.0
        }});
    return n_zero / len;
}


fn read_array_data(p: &str, has_header: bool, rows: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    println!("{}", &p);
    let file = File::open(p)?;  
    let mut reader = ReaderBuilder::new()
        .has_headers(has_header)
        .from_reader(file);
    let array_read: Array2<f64> = reader.deserialize_array2((rows, 785))?;
    Ok(array_read)
}


fn run_network() {
    let train_columns = 10000;
    let test_columns = 141;

    // use the test data to train because the train to big to load
    let train = read_array_data("/home/benja/projects/rust_2/mnist_test.csv", true, train_columns).unwrap();
    let test = read_array_data("/home/benja/projects/rust_2/test.csv", true, test_columns).unwrap();
    
    let x = &train.t();
    let y = &x.slice_axis( Axis(0), Slice::from(0..1));
    let x = &x.slice_axis( Axis(0), Slice::from(1..785))  / 255.0; 
    let y_train = Array::from_iter(y.iter().map(|&val| val as i32));

    let x_t = &test.t();
    let y_t = &x_t.slice_axis( Axis(0), Slice::from(0..1));
    let x_t = &x_t.slice_axis( Axis(0), Slice::from(1..785))  / 255.0; 
    let y_test = Array::from_iter(y_t.iter().map(|&val| val as i32));

    println!("\n{}", &y_train);
    println!("\n{}", &x);
    println!("\n{}", &y_test);
    println!("\n{}", &x);


    let (w1,b1,w2,b2) = gradient_descent(&x, &y_train, 10, 0.2);

    let (z1,a1,z2,a2) = foward_prop(&x,&w1,&b1,&w2,&b2);

    let y_pred = predictions(a2);

    let b1_test = b1.slice_axis( Axis(1), Slice::from(0..test_columns)).to_owned() ;
    let b2_test = b2.slice_axis( Axis(1), Slice::from(0..test_columns)).to_owned() ;

    let (z17,a1t,z2t,a2t) = foward_prop(&x_t,&w1,&b1_test,&w2,&b2_test);

    let y_pred_test = predictions(a2t);

    let acc = accuracy(&y_pred, &y_train);
    let acc_test = accuracy(&y_pred_test, &y_test);
    println!("accuracy  on trained dataset :  {}", &acc);
    println!("accuracy  on test dataset :  {}", &acc_test);

    // let soft_test = arr2(&[[100.0,2.9,4.3], [200.0,3.9,4.3], [1.0,300.9,9.3],[1.0,3.9,400.3],[1.0,7.9,4.3]]);
    // println!("{}", &soft_test);

    // let ss = softmax(&soft_test);
    // println!("{}", &ss);
}
