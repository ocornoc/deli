use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deli::{Triangle, Tetrahedron, Collidable};
use nalgebra::{Point4, Vector4, RealField};
use rand::distributions::{Standard, Distribution};

pub fn huge_collisions_triangles<N: RealField>(
    c: &mut Criterion
) where Standard: Distribution<N> {
    const NUMBER_OF_TRIS: usize = 1000000;
    let mut tris: Vec<Triangle<N>> = Vec::with_capacity(NUMBER_OF_TRIS);
    let rand_point: Point4<N> = Point4::from(Vector4::new_random());

    for _ in 0..NUMBER_OF_TRIS {
        tris.push(Triangle::new_random())
    };
    
    c.bench_function("Huge point-triangle collision test", |b| b.iter(||
        for tri in tris.iter() {
            tri.intersects(black_box(&rand_point));
        }
    ));
}

criterion_group!(
    huge_point_collisions,
    huge_collisions_triangles<f32>,
    huge_collisions_triangles<f64>);
criterion_main!(huge_point_collisions);