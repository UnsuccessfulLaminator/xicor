use ordered_float::OrderedFloat;
use num_traits::float::FloatCore;



/// Calculate the normalised xi-correlation of two floating-point sequences.
///
/// See [`xicor_norm`] for details of normalisation.
///
/// # Example
///
/// ```
/// use xicor::xicorf_norm;
///
/// let x: Vec<f32> = (0..47).map(|i| i as f32).collect();
/// let y: Vec<f32> = x.iter().map(|x| x*x).collect();
/// let xi = xicorf_norm(&x, &y);
///
/// assert_eq!(xi, 1.);
/// ```
///
/// Note that this is exactly the same data used in the example for [`xicorf`],
/// but here the result is actually 1.
pub fn xicorf_norm<F: FloatCore>(x: &[F], y: &[F]) -> f64 {
    let n = x.len() as f64;
    let lim = (n-2.)/(n+1.);

    xicorf(x, y)/lim
}

/// Calculate the normalised xi-correlation of two sequences whose values are
/// orderable (they implement [`Ord`]).
///
/// One particular quirk of Chatterjee's Xi is that its maximum value is only 1
/// in the limit as the number of points goes to infinity. For a finite number
/// of points `n`, its upper limit is `(n-2)/(n+1)`, _even with perfect
/// correlation_. This can be substantially below 1 for small datasets.
/// Therefore, this function is provided which normalises by that limit, such
/// that it will always return 1 for perfectly correlated data.
///
/// # Example
///
/// ```
/// use xicor::xicor_norm;
///
/// let x: Vec<u32> = (0..47).collect();
/// let y: Vec<u32> = x.iter().map(|x| x*x).collect();
/// let xi = xicor_norm(&x, &y);
///
/// assert_eq!(xi, 1.);
/// ```
///
/// Note that this is exactly the same data used in the example for [`xicor`],
/// but here the result is actually 1.
pub fn xicor_norm<T: Ord + Copy>(x: &[T], y: &[T]) -> f64 {
    let n = x.len() as f64;
    let lim = (n-2.)/(n+1.);

    xicor(x, y)/lim
}

/// Calculate the xi-correlation of two floating-point sequences.
///
/// This is a thin wrapper around [`xicor`] that transmutes slices of floats
/// into slices of [`OrderedFloat`], which implements the necessary [`Ord`].
///
/// # Example
///
/// ```
/// use xicor::xicorf;
///
/// let x: Vec<f32> = (0..47).map(|i| i as f32).collect();
/// let y: Vec<f32> = x.iter().map(|x| x*x).collect();
/// let xi = xicorf(&x, &y);
///
/// assert_eq!(xi, 0.9375);
/// ```
pub fn xicorf<F: FloatCore>(x: &[F], y: &[F]) -> f64 {
    // This is safe because OrderedFloat has transparent representation
    let x: &[OrderedFloat<F>] = unsafe { std::mem::transmute(x) };
    let y: &[OrderedFloat<F>] = unsafe { std::mem::transmute(y) };

    xicor(x, y)
}

/// Calculate the xi-correlation of two sequences whose values are orderable
/// (they implement [`Ord`]).
///
/// # Example
///
/// ```
/// use xicor::xicor;
///
/// let x: Vec<u32> = (0..47).collect();
/// let y: Vec<u32> = x.iter().map(|x| x*x).collect();
/// let xi = xicor(&x, &y);
///
/// assert_eq!(xi, 0.9375);
/// ```
pub fn xicor<T: Ord + Copy>(x: &[T], y: &[T]) -> f64 {
    assert!(x.len() == y.len(), "x and y must have the same length");

    let idcs = argsort(x);
    let y_ord = permute(y, &idcs);

    let idcs = argsort(&y_ord);
    let y_ascending = permute(&y_ord, &idcs);
    let r_ascending = cumulative_lte(&y_ascending);
    let l_ascending = cumulative_gte(&y_ascending);
    let mut rs = vec![0.; x.len()];
    let mut ls = vec![0.; x.len()];

    for ((i, r), l) in idcs.into_iter().zip(r_ascending).zip(l_ascending) {
        rs[i] = r as f64;
        ls[i] = l as f64;
    }

    let rsum = rs.windows(2)
        .map(|win| (win[0]-win[1]).abs())
        .sum::<f64>();

    let n = x.len() as f64;
    let lsum = ls.into_iter()
        .map(|l| l*(n-l))
        .sum::<f64>();

    1.-n*rsum/(2.*lsum)
}

// Return the indices that would sort the given array. That is, if you map the
// returned sequence of indices i -> arr[i], the resulting sequence is sorted.
pub(super) fn argsort<T: Ord>(arr: &[T]) -> Vec<usize> {
    let mut idcs: Vec<usize> = (0..arr.len()).collect();

    idcs.sort_unstable_by_key(|&i| &arr[i]);
    idcs
}

// Permute the given array such that arr[n] ends up at idcs[n].
pub(super) fn permute<T: Copy>(arr: &[T], idcs: &[usize]) -> Vec<T> {
    idcs.iter()
        .map(|&i| arr[i])
        .collect()
}

// For every element in the array, count how many elements are less than or
// equal to it. The array should be sorted before it is passed in.
pub(super) fn cumulative_lte<T: PartialEq<T> + Copy>(arr: &[T]) -> Vec<usize> {
    let mut counts: Vec<usize> = (1..=arr.len()).collect();

    for i in (0..arr.len()-1).rev() {
        if arr[i] == arr[i+1] { counts[i] = counts[i+1]; }
    }

    counts
}

// For every element in the array, count how many elements are greater than or
// equal to it. The array should be sorted before it is passed in.
pub(super) fn cumulative_gte<T: PartialEq<T> + Copy>(arr: &[T]) -> Vec<usize> {
    let mut counts: Vec<usize> = (1..=arr.len()).rev().collect();

    for i in 0..arr.len()-1 {
        if arr[i+1] == arr[i] { counts[i+1] = counts[i]; }
    }

    counts
}
