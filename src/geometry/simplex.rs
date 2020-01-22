use nalgebra::{
    Point4, Vector4, Point3, Point5,
    Scalar, RealField,
    SquareMatrix};
use super::traits::{Measure, Degenerable, Decomposable, ArealCoordinates};
use approx::{abs_diff_eq, relative_eq, ulps_eq, AbsDiffEq, RelativeEq, UlpsEq};
use rand::distributions::{Standard, Distribution};

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Triangle<N: Scalar> {
    pub verts: [Point4<N>; 3]
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Tetrahedron<N: Scalar> {
    pub verts: [Point4<N>; 4]
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Pentachoron<N: Scalar> {
    pub verts: [Point4<N>; 5]
}

impl<N: RealField> Measure for Triangle<N> {
    type Output = N;
    
    fn measure_squared(&self) -> Self::Output {
        // Cayley-Menger determinant, expanded.
        // 1/(2!^2 * 2^2) = 1/(2*2 * 4) = 1/16
        const SCALE: f64 = 1f64 / 16f64;
        let scalen = N::from_subset(&SCALE);
        let d01 = (self.verts[0] - self.verts[1]).norm_squared();
        let d02 = (self.verts[0] - self.verts[2]).norm_squared();
        let d12 = (self.verts[1] - self.verts[2]).norm_squared();
        let x = (d02 + d12) * d01;
        scalen * ((x + x) - (d02 - d12).powi(2) - d01.powi(2)).abs()
    }
}

impl<N: RealField> Measure for Tetrahedron<N> {
    type Output = N;
    
    /// Cayley-Menger determinant
    fn measure_squared(&self) -> Self::Output {
        use nalgebra::U5;
        // 1/(3!^2 * 2^3) = 1/(6^2 * 8) = 1/288
        const SCALE: f64 = 1f64 / 288f64;
        let d01 = (self.verts[0] - self.verts[1]).norm_squared();
        let d02 = (self.verts[0] - self.verts[2]).norm_squared();
        let d03 = (self.verts[0] - self.verts[3]).norm_squared();
        let d12 = (self.verts[1] - self.verts[2]).norm_squared();
        let d13 = (self.verts[1] - self.verts[3]).norm_squared();
        let d23 = (self.verts[2] - self.verts[3]).norm_squared();
        let mat = SquareMatrix::<N, U5, _>::new(
            N::zero(), d01,       d02,       d03,       N::one(),
            d01,       N::zero(), d12,       d13,       N::one(),
            d02,       d12,       N::zero(), d23,       N::one(),
            d03,       d13,       d23,       N::zero(), N::one(),
            N::one(),  N::one(),  N::one(),  N::one(),  N::zero()
        );
        (N::from_subset(&SCALE) * mat.determinant()).abs()
    }
}

impl<N: RealField> Measure for Pentachoron<N> {
    type Output = N;
    
    /// Cayley-Menger determinant
    fn measure_squared(&self) -> Self::Output {
        use nalgebra::U6;
        // 1/(4!^2 * 2^4) = 1/(24*24 * 16) = 1/9216
        const SCALE: f64 = 1f64 / 9216f64;
        let d01 = (self.verts[0] - self.verts[1]).norm_squared();
        let d02 = (self.verts[0] - self.verts[2]).norm_squared();
        let d03 = (self.verts[0] - self.verts[3]).norm_squared();
        let d04 = (self.verts[0] - self.verts[4]).norm_squared();
        let d12 = (self.verts[1] - self.verts[2]).norm_squared();
        let d13 = (self.verts[1] - self.verts[3]).norm_squared();
        let d14 = (self.verts[1] - self.verts[4]).norm_squared();
        let d23 = (self.verts[2] - self.verts[3]).norm_squared();
        let d24 = (self.verts[2] - self.verts[4]).norm_squared();
        let d34 = (self.verts[3] - self.verts[4]).norm_squared();
        let mat = SquareMatrix::<N, U6, _>::new(
            N::zero(), d01,       d02,       d03,       d04,       N::one(),
            d01,       N::zero(), d12,       d13,       d14,       N::one(),
            d02,       d12,       N::zero(), d23,       d24,       N::one(),
            d03,       d13,       d23,       N::zero(), d34,       N::one(),
            d04,       d14,       d24,       d34,       N::zero(), N::one(),
            N::one(),  N::one(),  N::one(),  N::one(),  N::one(),  N::zero()
        );
        (N::from_subset(&SCALE) * mat.determinant()).abs()
    }
}

impl<N: RealField> AbsDiffEq for Triangle<N> {
    type Epsilon = N::Epsilon;
    
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Self::Epsilon::default_epsilon()
    }
    
    fn abs_diff_eq(&self, rhs: &Self, epsilon: Self::Epsilon) -> bool {
        abs_diff_eq!(self.verts[0], rhs.verts[0], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[1], rhs.verts[1], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[2], rhs.verts[2], epsilon = epsilon)
    }
}

impl<N: RealField> AbsDiffEq for Tetrahedron<N> {
    type Epsilon = N::Epsilon;
    
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Self::Epsilon::default_epsilon()
    }
    
    fn abs_diff_eq(&self, rhs: &Self, epsilon: Self::Epsilon) -> bool {
        abs_diff_eq!(self.verts[0], rhs.verts[0], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[1], rhs.verts[1], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[2], rhs.verts[2], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[3], rhs.verts[3], epsilon = epsilon)
    }
}

impl<N: RealField> AbsDiffEq for Pentachoron<N> {
    type Epsilon = N::Epsilon;
    
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Self::Epsilon::default_epsilon()
    }
    
    fn abs_diff_eq(&self, rhs: &Self, epsilon: Self::Epsilon) -> bool {
        abs_diff_eq!(self.verts[0], rhs.verts[0], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[1], rhs.verts[1], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[2], rhs.verts[2], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[3], rhs.verts[3], epsilon = epsilon) &&
        abs_diff_eq!(self.verts[4], rhs.verts[4], epsilon = epsilon)
    }
}

impl<N: RealField> RelativeEq for Triangle<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    fn relative_eq(
        &self, rhs: &Self,
        epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool
    {
        relative_eq!(
            self.verts[0], rhs.verts[0],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[1], rhs.verts[1],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[2], rhs.verts[2],
            epsilon = epsilon, max_relative = max_relative)
    }
}

impl<N: RealField> RelativeEq for Tetrahedron<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    fn relative_eq(
        &self, rhs: &Self,
        epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool
    {
        relative_eq!(
            self.verts[0], rhs.verts[0],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[1], rhs.verts[1],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[2], rhs.verts[2],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[3], rhs.verts[3],
            epsilon = epsilon, max_relative = max_relative)
    }
}

impl<N: RealField> RelativeEq for Pentachoron<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    fn relative_eq(
        &self, rhs: &Self,
        epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool
    {
        relative_eq!(
            self.verts[0], rhs.verts[0],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[1], rhs.verts[1],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[2], rhs.verts[2],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[3], rhs.verts[3],
            epsilon = epsilon, max_relative = max_relative) &&
        relative_eq!(
            self.verts[4], rhs.verts[4],
            epsilon = epsilon, max_relative = max_relative)
    }
}

impl<N: RealField> UlpsEq for Triangle<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    fn ulps_eq(
        &self, rhs: &Self,
        epsilon: Self::Epsilon, max_ulps: u32) -> bool
    {
        ulps_eq!(
            self.verts[0], rhs.verts[0],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[1], rhs.verts[1],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[2], rhs.verts[2],
            epsilon = epsilon, max_ulps = max_ulps)
    }
}

impl<N: RealField> UlpsEq for Tetrahedron<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    fn ulps_eq(
        &self, rhs: &Self,
        epsilon: Self::Epsilon, max_ulps: u32) -> bool
    {
        ulps_eq!(
            self.verts[0], rhs.verts[0],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[1], rhs.verts[1],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[2], rhs.verts[2],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[3], rhs.verts[3],
            epsilon = epsilon, max_ulps = max_ulps)
    }
}

impl<N: RealField> UlpsEq for Pentachoron<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    fn ulps_eq(
        &self, rhs: &Self,
        epsilon: Self::Epsilon, max_ulps: u32) -> bool
    {
        ulps_eq!(
            self.verts[0], rhs.verts[0],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[1], rhs.verts[1],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[2], rhs.verts[2],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[3], rhs.verts[3],
            epsilon = epsilon, max_ulps = max_ulps) &&
        ulps_eq!(
            self.verts[4], rhs.verts[4],
            epsilon = epsilon, max_ulps = max_ulps)
    }
}

impl<N: RealField> Degenerable for Triangle<N> {
    /// Returns if the `Triangle` is point-like (ie, has nearly 0 measure).
    #[inline]
    fn is_degenerate(&self) -> bool {
        const DEGENERATE_CONTENT: f64 = 0.000001;
         
        let eps = N::from_subset(&DEGENERATE_CONTENT);
        abs_diff_eq!(self.measure_squared(), N::zero(), epsilon = eps)
    }
}

impl<N: RealField> Degenerable for Tetrahedron<N> {
    /// Returns if the `Tetrahedron` is point-like (ie, has nearly 0 measure).
    #[inline]
    fn is_degenerate(&self) -> bool {
        const DEGENERATE_CONTENT: f64 = 0.001;

        let eps = N::from_subset(&DEGENERATE_CONTENT);
        abs_diff_eq!(self.measure_squared(), N::zero(), epsilon = eps)
    }
}

impl<N: RealField> Degenerable for Pentachoron<N> {
    /// Returns if the `Pentachoron` is point-like (ie, has nearly 0 measure).
    #[inline]
    fn is_degenerate(&self) -> bool {
        const DEGENERATE_CONTENT: f64 = 0.001;

        let eps = N::from_subset(&DEGENERATE_CONTENT);
        abs_diff_eq!(self.measure_squared(), N::zero(), epsilon = eps)
    }
}

/// Get the points of a triangle.
impl<N: RealField> Decomposable<Point4<N>> for Triangle<N> {
    type Output = [Point4<N>; 3];
    /// Returns the points of a triangle.
    #[inline]
    fn decompose(&self) -> Self::Output {
        self.verts
    }
    /// Creates a triangle from its points.
    #[inline]
    fn recompose(components: &Self::Output) -> Option<Self> {
        Some(Self::of_pverts(
            components[0],
            components[1],
            components[2]))
    }
}

/// Get the line segments of a triangle.
impl<N: RealField> Decomposable<(Point4<N>, Point4<N>)> for Triangle<N> {
    type Output = [(Point4<N>, Point4<N>); 3];
    /// Returns the line segments making up the triangle.
    #[inline]
    fn decompose(&self) -> Self::Output {
        [(self.verts[0], self.verts[1]),
         (self.verts[1], self.verts[2]),
         (self.verts[2], self.verts[0])]
    }
    /// Creates a triangle from its line segments.
    #[inline]
    fn recompose(components: &Self::Output) -> Option<Self> {
        const SEGMENT_POINT_EPS: f64 = 0.000001;
        
        let speps = N::from_subset(&SEGMENT_POINT_EPS);
        let v0eqv0 = abs_diff_eq!(
            components[0].0,
            components[2].1,
            epsilon = speps);
        let v1eqv1 = abs_diff_eq!(
            components[0].0,
            components[2].1,
            epsilon = speps);
        let v2eqv2 = abs_diff_eq!(
            components[0].0,
            components[2].1,
            epsilon = speps);
        if v0eqv0 && v1eqv1 && v2eqv2 {
            Some(Triangle::of_pverts(
                components[0].0,
                components[1].0,
                components[2].0
            ))
        } else { None }
    }
}

impl<N: RealField> ArealCoordinates for Triangle<N> {
    type Output = Option<Point3<N>>;
    type PointType = Point4<N>;
    
    /// Returns `None` if this triangle is degenerate or `p` isn't coplanar to
    /// the triangle. Otherwise, returns the areal coordinates, where the first
    /// second, ... coordinates map to the first, second, ... vertices.
    fn get_areal_of_cart(&self, p: &Self::PointType) -> Self::Output {
        // If the triangle is degenerate, then the point approximately won't
        // intersect. If it isn't degenerate, then make sure `p` is coplanar.
        if self.is_degenerate() || !self.is_point_coplanar(p) { return None }
        // XXX https://bit.ly/35DZtcZ thank you user2357 XXX
        use nalgebra::Matrix4x3;
        let mat = Matrix4x3::from_columns(&[
            self.verts[0].coords, self.verts[1].coords, self.verts[2].coords
        ]).insert_row(0, N::one());
        let mt = mat.transpose();
        if let Some(mat2) = (mt * mat).try_inverse() {
            Some(Point3::from((mat2 * mt) * p.coords.insert_row(0, N::one())))
        } else { None }
    }
}

impl<N: RealField> ArealCoordinates for Tetrahedron<N> {
    type Output = Option<Point4<N>>;
    type PointType = Point4<N>;

    /// Returns `None` if this tetrahedron is degenerate or `p` isn't
    /// cohyperplanar to the tetrahedron. Otherwise, returns the areal
    /// coordinates, where the first, second, ... coordinates map to the first,
    /// second, ... vertices.
    fn get_areal_of_cart(&self, p: &Point4<N>) -> Option<Point4<N>> {
        // If the tetrahedron is degenerate, then the point approximately won't
        // intersect. If it isn't degenerate, then make sure `p` is
        // cohyperplanar.
        if self.is_degenerate() || !self.is_point_cohyplanar(p) { return None };
        // XXX https://bit.ly/35DZtcZ thank you user2357 XXX
        let mat = SquareMatrix::from_columns(&[
            self.verts[0].coords, self.verts[1].coords, self.verts[2].coords,
            self.verts[3].coords
        ]).insert_row(0, N::one());
        let mt = mat.transpose();
        if let Some(mat2) = (mt * mat).try_inverse() {
            Some(Point4::from((mat2 * mt) * p.coords.insert_row(0, N::one())))
        } else { None }
    }
}

impl<N: RealField> ArealCoordinates for Pentachoron<N> {
    type Output = Option<Point5<N>>;
    type PointType = Point4<N>;

    /// Returns `None` if this pentachoron is degenerate. Otherwise, returns
    /// the areal coordinates, where the first, second, ... coordinates map to
    /// the first, second, ... vertices.
    fn get_areal_of_cart(&self, p: &Point4<N>) -> Option<Point5<N>> {
        use nalgebra::Matrix4x5;
        // If the pentachoron is degenerate, then the point approximately won't
        // intersect.
        if self.is_degenerate() { return None };
        // XXX https://bit.ly/35DZtcZ thank you user2357 XXX
        let mat = Matrix4x5::from_columns(&[
            self.verts[0].coords, self.verts[1].coords, self.verts[2].coords,
            self.verts[3].coords, self.verts[4].coords
        ]).insert_row(0, N::one());
        if let Some(mat2) = mat.try_inverse() {
            Some(Point5::from(mat2 * p.coords.insert_row(0, N::one())))
        } else { None }
    }
}

impl<N: RealField> Triangle<N> {
    #[inline]
    pub fn of_pverts(p0: Point4<N>, p1: Point4<N>, p2: Point4<N>) -> Self {
        Self {verts: [p0, p1, p2]}
    }

    #[inline]
    pub fn new(p0: Point4<N>, p1: Point4<N>, p2: Point4<N>) -> Self {
        Self::of_pverts(p0, p1, p2)
    }

    #[inline]
    pub fn of_vverts(v0: Vector4<N>, v1: Vector4<N>, v2: Vector4<N>) -> Self {
        Self::of_pverts(Point4::from(v0), Point4::from(v1), Point4::from(v2))
    }

    /// Essentially, if a tetrahedron made of this triangle and the point is
    /// degenerate and the triangle itself isn't degenerate, then the point and
    /// the triangle must be coplanar. Doesn't test if this triangle is
    /// degenerate.
    #[inline]
    pub fn is_point_coplanar(&self, p: &Point4<N>) -> bool {
        Tetrahedron::of_tri_point(self, *p).is_degenerate()
    }

    /// Creates a new random simplex.
    #[inline]
    pub fn new_random() -> Self
    where Standard: Distribution<N>
    {
        Self::of_vverts(
            Vector4::new_random(),
            Vector4::new_random(),
            Vector4::new_random()
        )
    }
}

impl<N: RealField> Tetrahedron<N> {
    #[inline]
    pub fn of_pverts(
        p0: Point4<N>, p1: Point4<N>,
        p2: Point4<N>, p3: Point4<N>) -> Self
    {
        Self {verts: [p0, p1, p2, p3]}
    }

    #[inline]
    pub fn new(
        p0: Point4<N>, p1: Point4<N>,
        p2: Point4<N>, p3: Point4<N>) -> Self
    {
        Self::of_pverts(p0, p1, p2, p3)
    }

    #[inline]
    pub fn of_vverts(
        v0: Vector4<N>, v1: Vector4<N>,
        v2: Vector4<N>, v3: Vector4<N>) -> Self
    {
        Self::of_pverts(
            Point4::from(v0), Point4::from(v1),
            Point4::from(v2), Point4::from(v3))
    }

    #[inline]
    pub fn of_tri_point(tri: &Triangle<N>, p: Point4<N>) -> Self {
        Self::of_pverts(tri.verts[0], tri.verts[1], tri.verts[2], p)
    }

    /// Essentially, if a pentachoron made of this triangle and the point is
    /// degenerate and the tetrahedron itself isn't degenerate, then the point
    /// and the tetrahedron must be cohyperplanar. Doesn't test if this
    /// tetrahedron is degenerate.
    #[inline]
    pub fn is_point_cohyplanar(&self, p: &Point4<N>) -> bool {
        Pentachoron::of_tet_point(self, *p).is_degenerate()
    }

    /// Creates a new random simplex.
    #[inline]
    pub fn new_random() -> Self
    where Standard: Distribution<N>
    {
        Self::of_vverts(
            Vector4::new_random(),
            Vector4::new_random(),
            Vector4::new_random(),
            Vector4::new_random()
        )
    }
}

impl<N: RealField> Pentachoron<N> {
    #[inline]
    pub fn of_pverts(
        p0: Point4<N>, p1: Point4<N>, p2: Point4<N>,
        p3: Point4<N>, p4: Point4<N>) -> Self
    {
        Self {verts: [p0, p1, p2, p3, p4]}
    }

    #[inline]
    pub fn new(
        p0: Point4<N>, p1: Point4<N>, p2: Point4<N>,
        p3: Point4<N>, p4: Point4<N>) -> Self
    {
        Self::of_pverts(p0, p1, p2, p3, p4)
    }

    #[inline]
    pub fn of_vverts(
        v0: Vector4<N>, v1: Vector4<N>, v2: Vector4<N>,
        v3: Vector4<N>, v4: Vector4<N>) -> Self
    {
        Self::of_pverts(
            Point4::from(v0), Point4::from(v1), Point4::from(v2),
            Point4::from(v3), Point4::from(v4))
    }

    #[inline]
    pub fn of_tri_point_point(
        tri: &Triangle<N>,
        p0: Point4<N>, p1: Point4<N>) -> Self
    {
        Self::of_pverts(tri.verts[0], tri.verts[1], tri.verts[2], p0, p1)
    }

    #[inline]
    pub fn of_tet_point(tret: &Tetrahedron<N>, p: Point4<N>) -> Self {
        Self::of_pverts(
            tret.verts[0], tret.verts[1],
            tret.verts[2], tret.verts[3], p)
    }

    /// Creates a new random simplex.
    #[inline]
    pub fn new_random() -> Self
    where Standard: Distribution<N>
    {
        Self::of_vverts(
            Vector4::new_random(),
            Vector4::new_random(),
            Vector4::new_random(),
            Vector4::new_random(),
            Vector4::new_random()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use approx::assert_abs_diff_eq;

    #[test]
    fn triangle_measure_test_zero() {
        let measure_tri = vec![
            (0f32, Triangle::new(
                Point4::origin(),
                Point4::origin(),
                Point4::origin()
            )),
            (0f32, Triangle::new(
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::origin(),
                Point4::origin()
            )),
            (0f32, Triangle::new(
                Point4::origin(),
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::origin()
            )),
            (0f32, Triangle::new(
                Point4::origin(),
                Point4::origin(),
                Point4::new(1.0, 1.0, 1.0, 1.0),
            )),
            (0f32, Triangle::new(
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::origin()
            )),
            (0f32, Triangle::new(
                Point4::origin(),
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::new(1.0, 1.0, 1.0, 1.0),
            )),
            (0f32, Triangle::new(
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::new(1.0, 1.0, 1.0, 1.0),
                Point4::new(1.0, 1.0, 1.0, 1.0),
            )),
            (0f32, Triangle::new(
                Point4::new(1.0, 2.0, 3.0, 0.0),
                Point4::new(3.0, 2.0, 1.0, 0.0),
                Point4::new(2.0, 2.0, 2.0, 0.0),
            )),
        ];

        for (m, p) in measure_tri {
            assert_eq!(m, p.measure())
        }
    }

    #[test]
    fn triangle_measure_test_nonzero() {
        let measure_tri = vec![
            (5f32.sqrt(), Triangle::new(
                Point4::new(1.0, 2.0, 3.0, 4.0),
                Point4::new(4.0, 3.0, 2.0, 1.0),
                Point4::new(2.0, 2.0, 2.0, 2.0),
            )),
            (38.531895, Triangle::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
            ))
        ];

        for (m, p) in measure_tri {
            assert_abs_diff_eq!(m, p.measure(), epsilon = 0.000001)
        }
    }

    #[test]
    fn triangle_2d_coplanar() {
        let tri = Triangle::<f32>::new(
            Point4::new(-2.6, 0.2, 0.0, 0.0),
            Point4::new(-1.1, 3.1, 0.0, 0.0),
            Point4::new(-6.8, 1.8, 0.0, 0.0)
        );
        let points_cop = vec![
            Point4::new(-1.6, 2.7, 0.0, 0.0),
            Point4::new(-3.5, 1.7, 0.0, 0.0),
            Point4::new(-4.8, 4.7, 0.0, 0.0)
        ];
        let points_not_cop = vec![
            Point4::new(-1.6, 2.7, 1.0, 0.0),
            Point4::new(-3.5, 1.7, 0.0, 1.0),
            Point4::new(-4.8, 4.7, 0.1, 0.1)
        ];

        for p in points_cop {
            assert!(tri.is_point_coplanar(&p));
        }

        for p in points_not_cop {
            assert!(!tri.is_point_coplanar(&p));
        }
    }

    #[test]
    fn triangle_areal() {
        use nalgebra::Point3;

        let tri = Triangle::<f32>::new(
            Point4::new(3.5, 5.5, -7.0, -2.0),
            Point4::new(0.0, 1.0, -1.0, 0.0),
            Point4::new(-2.1, 5.0, 4.0, -6.1),
        );
        let points_areal = vec![
            (tri.verts[0], Point3::new(1.0, 0.0, 0.0)),
            (tri.verts[1], Point3::new(0.0, 1.0, 0.0)),
            (tri.verts[2], Point3::new(0.0, 0.0, 1.0)),
            (Point4::new(0.466667, 3.83333, -1.33333, -2.7),
                Point3::new(0.33333, 0.33333, 0.33333)),
        ];

        for (pc, pa) in points_areal {
            let tria = tri.get_areal_of_cart(&pc)
                .expect("Failed to get areal coordinates in test");
            let areals = tria.x + tria.y + tria.z;
            assert_abs_diff_eq!(areals, 1.0, epsilon = 0.0001);
            assert_abs_diff_eq!((pa - tria).norm(), 0.0, epsilon = 0.0001);
        }
    }
}