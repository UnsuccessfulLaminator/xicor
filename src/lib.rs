use ordered_float::OrderedFloat;
use num_traits::float::FloatCore;



/// Calculate the xi-correlation of two floating-point sequences.
pub fn xicorf<F: FloatCore>(x: &[F], y: &[F]) -> f64 {
    // This is safe because OrderedFloat has transparent representation
    let x: &[OrderedFloat<F>] = unsafe { std::mem::transmute(x) };
    let y: &[OrderedFloat<F>] = unsafe { std::mem::transmute(y) };

    xicor(x, y)
}

/// Calculate the xi-correlation of two sequences whose values are orderable
/// (they implement `Ord`).
pub fn xicor<T: Ord + Copy>(x: &[T], y: &[T]) -> f64 {
    assert!(x.len() == y.len(), "x and y must have the same length");

    let idcs = argsort(x);
    let y_ord = permute(y, &idcs);

    let idcs = argsort(&y_ord);
    let y_ascending = permute(&y_ord, &idcs);
    let r_ascending = cumulative_count(&y_ascending);
    let mut rs = vec![0; x.len()];

    for (i, r) in idcs.into_iter().zip(r_ascending) {
        rs[i] = r;
    }

    let n = x.len() as f64;
    let rsum = rs.windows(2)
        .map(|win| win[0].abs_diff(win[1]))
        .sum::<usize>();

    1.-3.*(rsum as f64)/(n*n-1.)
}

// Return the indices that would sort the given array. That is, if you map the
// returned sequence of indices i -> arr[i], the resulting sequence is sorted.
fn argsort<T: Ord>(arr: &[T]) -> Vec<usize> {
    let mut idcs: Vec<usize> = (0..arr.len()).collect();

    idcs.sort_unstable_by_key(|&i| &arr[i]);
    idcs
}

// Permute the given array such that arr[n] ends up at idcs[n].
fn permute<T: Copy>(arr: &[T], idcs: &[usize]) -> Vec<T> {
    idcs.iter()
        .map(|&i| arr[i])
        .collect()
}

// For every element in the array, count how many elements are less than or
// equal to it. The array should be sorted before it is passed in.
fn cumulative_count<T: PartialEq<T> + Copy>(arr: &[T]) -> Vec<usize> {
    let mut counts: Vec<usize> = vec![0; arr.len()];
    let mut prev = arr[arr.len()-1];
    let mut prev_count = arr.len();

    for (i, (c, &a)) in counts.iter_mut().zip(arr).enumerate().rev() {
        if a == prev { *c = prev_count; }
        else {
            *c = i+1;
            prev = a;
            prev_count = *c;
        }
    }

    counts
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xicor() {
        let x = [1, 4, -9, -6, -5, -8, -1, 0, -4, -5];
        let y = [9, 8, 5, -10, 7, -6, -2, -8, 4, 3];

        assert_eq!(xicor(&x, &y), 0.09090909090909094);
    }

    #[test]
    fn test_xicorf() {
        let x: Vec<f32> = (0..1000).map(|i| i as f32/1000.).collect();
        let y: Vec<f32> = x.iter().map(|&x| (x*12.566).sin()).collect();

        // There is strong correlation forwards, because y is a one-to-one
        // function of x
        assert_eq!(xicorf(&x, &y), 0.988032988032988);

        // There is only very weak correlation backwards, because x is not a
        // one-to-one function of y
        assert_eq!(xicorf(&y, &x), 0.06099306099306101);
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
    fn test_cumulative_count() {
        let arr = [1.1, 1.2, 2.5, 2.5, 2.5, 3., 8., 19., 19., 51.7, 100.];
        let counts = [1, 2, 5, 5, 5, 6, 7, 9, 9, 10, 11];

        assert_eq!(cumulative_count(&arr).as_slice(), &counts);
    }
}
