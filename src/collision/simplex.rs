use crate::collision::collidable::CollisionDescribable;
use crate::geometry::ArealCoordinates;
use nalgebra::{
    RealField,
    Point4, Point,
    DimName,
    DefaultAllocator, allocator::Allocator};

impl<N: RealField, D: DimName, T> CollisionDescribable<Point4<N>> for T
where T: ArealCoordinates<Output=Option<Point<N, D>>, PointType=Point4<N>>,
      DefaultAllocator: Allocator<N, D>
{
    type CollisionType = Option<Point4<N>>;
    
    /// Uses the areal coordinates of `p` to find if `p` is in `self`.
    fn get_collision(&self, p: &Point4<N>) -> Self::CollisionType {
        if let Some(areal) = self.get_areal_of_cart(p) {
            let all_between_zero_one = areal.iter().all(| comp |
                N::zero() < *comp && *comp < N::one()
            );

            if all_between_zero_one { Some(*p) } else { None }
        } else { None }
    }
}

/*impl<N: RealField> CollisionDescribable<Point4<N>> for Triangle<N> {
    type CollisionType = Option<Point4<N>>;
    
    /// Uses the areal coordinates of `p` to find if `p` is in `self`.
    fn get_collision(&self, p: &Point4<N>) -> Self::CollisionType {
        if let Some(areal) = self.get_areal(p) {
            let arealxinrange = N::zero() < areal.x && areal.x < N::one();
            let arealyinrange = N::zero() < areal.y && areal.y < N::one();
            let arealzinrange = N::zero() < areal.z && areal.z < N::one();

            if arealxinrange && arealyinrange && arealzinrange {
                Some(*p)
            } else { None }
        } else { None }
    }
}

#[inline]
fn npos<N: RealField>(n: &N) -> bool {
    use nalgebra::zero;
    
    zero::<N>() < *n
}

impl<N: RealField> CollisionDescribable<Point4<N>> for Tetrahedron<N> {
    type CollisionType = Option<Point4<N>>;
    
    fn get_collision(&self, p: &Point4<N>) -> Self::CollisionType {
        use nalgebra::U4;
        if self.is_degenerate() { return None };

        let mat0dspos = npos(&SquareMatrix::<N, U4, _>::from_columns(&[
            self.verts[0].coords,
            self.verts[1].coords,
            self.verts[2].coords,
            self.verts[3].coords]).determinant());
        let mat1dspos = npos(&SquareMatrix::<N, U4, _>::from_columns(&[
            p.coords,
            self.verts[1].coords,
            self.verts[2].coords,
            self.verts[3].coords]).determinant());
        let mat2dspos = npos(&SquareMatrix::<N, U4, _>::from_columns(&[
            self.verts[0].coords,
            p.coords,
            self.verts[2].coords,
            self.verts[3].coords]).determinant());
        let mat3dspos = npos(&SquareMatrix::<N, U4, _>::from_columns(&[
            self.verts[0].coords,
            self.verts[1].coords,
            p.coords,
            self.verts[3].coords]).determinant());
        let mat4dspos = npos(&SquareMatrix::<N, U4, _>::from_columns(&[
            self.verts[0].coords,
            self.verts[1].coords,
            self.verts[2].coords,
            p.coords]).determinant());
        let mut samesign = true;

        for x in &[mat1dspos, mat2dspos, mat3dspos, mat4dspos] {
            samesign &= mat0dspos == *x
        };
        
        if samesign { Some(*p) } else { None }
    }
}

impl<N: RealField> CollisionDescribable<Point4<N>> for Pentachoron<N> {
    type CollisionType = Option<Point4<N>>;
    
    fn get_collision(&self, p: &Point4<N>) -> Self::CollisionType {
        if let Some(areal) = self.get_areal(p) {
            let areals = areal.y + areal.z + areal.w + areal.a;

            if N::zero() < areals && areals < N::one() {
                Some(*p)
            } else { None }
        } else { None }
    }
}*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::collidable::Collidable;
    use crate::geometry::{Triangle};

    #[test]
    fn triangle_2d_collision_test() {
        let tri = Triangle::<f32>::new(
            Point4::new(-2.6, 0.2, 0.0, 0.0),
            Point4::new(-1.1, 3.1, 0.0, 0.0),
            Point4::new(-6.8, 1.8, 0.0, 0.0)
        );
        let points_in = vec![
            Point4::new(-3.5, 1.7, 0.0, 0.0),
        ];
        let points_out = vec![
            Point4::new(-3.5, 1.7, 1.0, 0.0),
            Point4::new(-3.5, 1.7, 0.0, 1.0),
            Point4::new(-3.5, 1.7, 1.0, 1.0),
            Point4::new(-3.5, 1.7, -1.0, 0.0),
            Point4::new(-3.5, 1.7, 0.0, -1.0),
            Point4::new(-3.5, 1.7, -1.0, -1.0),
            Point4::new(-4.8, 4.7, 0.0, 0.0)
        ];
        
        for p in points_in {
            assert!(tri.intersects(&p))
        }
        
        for p in points_out {
            assert!(!tri.intersects(&p))
        }
    }
}