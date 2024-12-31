use super::*;



// Fractional tolerance for testing approximate equality of floats
const RTOL: f64 = 1e-8;

// Check whether 2 floating-point values are within the given fractional
// tolerance of eachother.
fn assert_req(left: f64, right: f64, rtol: f64) {
    let mean_size = (left.abs()+right.abs())/2.;
    let diff = (left-right).abs();
    
    if diff/mean_size > rtol {
        panic!(
            "assertion `left ~= right` failed\n\
            left: {left}\n\
            right: {right}"
        );
    }
}



#[test]
fn test_xicor() {
    let x = [1, 4, -9, -6, -5, -8, -1, 0, -4, -5];
    let y = [9, 8, 5, -10, 7, -6, -2, -8, 4, 3];

    assert_req(xicor(&x, &y), 0.0909090909, RTOL);
}

#[test]
fn test_xicorf() {
    let x: Vec<f32> = (0..1000).map(|i| i as f32/1000.).collect();
    let y: Vec<f32> = x.iter().map(|&x| (x*12.566).sin()).collect();

    // There is strong correlation forwards, because y is a function of x
    assert_req(xicorf(&x, &y), 0.9880330596, RTOL);

    // There is only very weak correlation backwards, because x is not a
    // function of y - this is a one-to-many relationship
    assert_req(xicorf(&y, &x), 0.0609930609, RTOL);
}

#[test]
fn test_xicor_norm() {
    let x = [1, 4, -9, -6, -5, -8, -1, 0, -4, -5];
    let y = [9, 8, 5, -10, 7, -6, -2, -8, 4, 3];

    assert_req(xicor_norm(&x, &y), 0.125, RTOL);
}

#[test]
fn test_xicorf_norm() {
    let x: Vec<f32> = (0..1000).map(|i| i as f32/1000.).collect();
    let y: Vec<f32> = x.iter().map(|&x| (x*12.566).sin()).collect();

    assert_req(xicorf_norm(&x, &y), 0.9910030989, RTOL);
}

#[test]
fn test_argsort() {
    let arr = [2, -2, -9, 8, 4, 1, 6, -3];
    let idcs = [2, 7, 1, 5, 0, 4, 6, 3];

    assert_eq!(argsort(&arr).as_slice(), &idcs);
}

#[test]
fn test_permute() {
    let arr = [2, -2, -9, 8, 4, 1, 6, -3];
    let idcs = [2, 7, 1, 5, 0, 4, 6, 3];
    let permuted = [-9, -3, -2, 1, 2, 4, 6, 8];

    assert_eq!(permute(&arr, &idcs).as_slice(), &permuted);
}

#[test]
fn test_cumulative_lte() {
    let arr = [1.1, 1.1, 2.5, 2.5, 2.5, 3., 8., 19., 19., 51.7, 51.7];
    let counts = [2, 2, 5, 5, 5, 6, 7, 9, 9, 11, 11];

    assert_eq!(cumulative_lte(&arr).as_slice(), &counts);
}

#[test]
fn test_cumulative_gte() {
    let arr = [1.1, 1.1, 2.5, 2.5, 2.5, 3., 8., 19., 19., 51.7, 100.];
    let counts = [11, 11, 9, 9, 9, 6, 5, 4, 4, 2, 1];

    assert_eq!(cumulative_gte(&arr).as_slice(), &counts);
}
