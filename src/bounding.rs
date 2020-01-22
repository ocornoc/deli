use nalgebra as na;
use na::{allocator::Allocator, RealField, DimName, Point, VectorN, DefaultAllocator};

/// Given a vector of `D`-dimensional points in `N`, returns the centroid.
/// When no points, returns the origin. Safe to use with INF and NaNs.
pub fn get_centroid<N, D>(points: &Vec<Point<N, D>>) -> Point<N, D>
    where N: RealField,
          D: DimName,
          DefaultAllocator: Allocator<N, D>
{
    let lenrecip: N = na::one::<N>() / na::convert(points.len() as f64);

    points.iter().fold(Point::origin(), |l, r| l + (r.coords.scale(lenrecip)))
}

/// Given a vector of `D`-dimensional points in `N`, returns a hypersphere
/// encompassing all the points. The first tuple value is the radius, and the
/// second is the center of the hypersphere.
pub fn bounding_nsphere<N, D>(points: &Vec<Point<N, D>>) -> (N, Point<N, D>)
    where N: RealField,
          D: DimName,
          DefaultAllocator: Allocator<N, D>
{
    let center = get_centroid(points);
    let maxradius: Option<N> = points.iter()
        .map(|p| na::distance(p, &center))
        .max_by(|x, y|
            match x.partial_cmp(y) {
                None      => std::cmp::Ordering::Less,
                Some(ord) => ord
            }
        );

    (if let Some(maxr) = maxradius { maxr } else { na::zero() }, center)
}

fn basis_vs<N, D>() -> Vec<VectorN<N, D>>
    where N: RealField,
          D: DimName,
          DefaultAllocator: Allocator<N, D>
{
    debug_assert!(D::dim() >= 1, "The dimension must be at least 1 to get basis vectors!");
    let dimsub1 = D::dim() - 1;

    (0..dimsub1)
        .collect::<Vec<_>>()
        .iter()
        .map(|x| VectorN::<N, D>::from_fn(|i, _| na::convert((i == *x) as u8 as f64)))
        .collect::<Vec<VectorN<N, D>>>()
}

/// Given a vector of `D`-dimensional points in `N`, returns a `D`-simplex encompassing
/// all the points. The vector returned is the vertices of the simplex, and should have
/// `D` length. Assumes `D >= 2` and will debug-assert that to be the case.
pub fn bounding_simplex<N, D>(points: &Vec<Point<N, D>>) -> Vec<Point<N, D>>
    where N: RealField,
          D: DimName,
          DefaultAllocator: Allocator<N, D>
{
    debug_assert!(D::dim() >= 2, "The dimension must be at least 2!");

    let bsphere = bounding_nsphere(points);
    // 6.0 * -(2.0 + 5.0f64.sqrt()). c'mon rust, get const sqrt already!
    const FACTOR: f64 = -25.41640786499873817;
    let mut simplexverts = basis_vs().iter()
        .map(|v| v.scale(na::convert::<_, N>(3.0) * bsphere.0) + bsphere.1.clone().coords)
        .collect::<Vec<VectorN<N, D>>>();
    
    if let Some(fin) = simplexverts.last_mut() {
        *fin = VectorN::<N, D>::repeat(bsphere.0 * na::convert(FACTOR)) + bsphere.1.coords;
    }
    
    simplexverts.iter()
        .map(|v| Point::<N, D>::from(v.clone()))
        .collect()
}


// Unit tests

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    // No panic when getting the centroid of U0 points.
    #[test]
    fn get_centroid_test_u0() {
        let data: Vec<na::Point<f32, na::dimension::U0>> = vec![
            na::Point::origin(),
            na::Point::origin(),
            na::Point::origin(),
            na::Point::origin()
        ];
        
        assert_eq!(get_centroid(&data), na::Point::origin());
    }

    // No panic when getting the centroid of an empty set of U0 points.
    #[test]
    fn get_centroid_test_u0_empty() {
        let data: Vec<na::Point<f32, na::dimension::U0>> = vec![];
        
        assert_eq!(get_centroid(&data), na::Point::origin());
    }

    // No panic when getting the centroid of U1 points.
    #[test]
    fn get_centroid_test_u1() {
        let data: Vec<na::Point1<f64>> = vec![
            na::Point1::new(1.0),
            na::Point1::new(-2.0),
            na::Point1::new(3.0),
            na::Point1::origin()
        ];
        
        assert_eq!(get_centroid(&data), na::Point1::new(0.5));
    }

    // Centroid of empty set of U1 is origin.
    #[test]
    fn get_centroid_test_u1_empty() {
        let data: Vec<na::Point1<f32>> = vec![];
        
        assert_eq!(get_centroid(&data), na::Point1::origin());
    }

    // NaN in any component will make the centroid's component NaN.
    #[test]
    fn get_centroid_test_nan() {
        let data: Vec<na::Point1<f64>> = vec![
            na::Point1::new(std::f64::NAN),
            na::Point1::new(-2.0),
            na::Point1::new(3.0),
            na::Point1::origin()
        ];
        
        let centroid = get_centroid(&data);
        assert_ne!(centroid, centroid);
    }

    // INF in any component will make the centroid's component INF.
    #[test]
    fn get_centroid_test_inf() {
        let data: Vec<na::Point1<f32>> = vec![
            na::Point1::new(std::f32::INFINITY),
            na::Point1::new(-2.0),
            na::Point1::new(3.0),
            na::Point1::origin()
        ];
        
        assert_eq!(get_centroid(&data), na::Point1::new(std::f32::INFINITY));
    }

    // No panic when getting the centroid of U3 points.
    #[test]
    fn get_centroid_test_u3() {
        let data: Vec<na::Point3<f32>> = vec![
            na::Point3::new(1.0, 3.5, 10.0),
            na::Point3::new(-2.0, 0.0, -2.0),
            na::Point3::new(1.0, -0.5, 1.0),
            na::Point3::origin()
        ];
        
        assert_eq!(get_centroid(&data), na::Point3::new(0.0, 0.75, 2.25));
    }

    // Centroid of empty set of U3 is origin.
    #[test]
    fn get_centroid_test_u3_empty() {
        let data: Vec<na::Point3<f64>> = vec![];
        
        assert_eq!(get_centroid(&data), na::Point3::origin());
    }

    // No panic when getting the bounding sphere of U0 points.
    #[test]
    fn bounding_nsphere_test_u0() {
        let data: Vec<na::Point<f32, na::dimension::U0>> = vec![
            na::Point::origin(),
            na::Point::origin(),
            na::Point::origin(),
            na::Point::origin()
        ];
        
        assert_eq!(bounding_nsphere(&data), (0.0, na::Point::origin()));
    }

    // No panic when getting the bounding sphere of an empty set of U0 points.
    #[test]
    fn bounding_nsphere_test_u0_empty() {
        let data: Vec<na::Point<f32, na::dimension::U0>> = vec![];
        
        assert_eq!(bounding_nsphere(&data), (0.0, na::Point::origin()));
    }

    // No panic when getting the bounding sphere of U1 points.
    #[test]
    fn bounding_nsphere_test_u1() {
        let data: Vec<na::Point1<f64>> = vec![
            na::Point1::new(0.1),
            na::Point1::new(10.0),
            na::Point1::new(-5.0),
            na::Point1::origin()
        ];
        
        assert_eq!(bounding_nsphere(&data), (8.725, na::Point1::new(1.275)));
    }

    // No panic when getting the bounding sphere of an empty set of U1 points.
    #[test]
    fn bounding_nsphere_test_u1_empty() {
        let data: Vec<na::Point1<f64>> = vec![];
        
        assert_eq!(bounding_nsphere(&data), (0.0, na::Point::origin()));
    }
}