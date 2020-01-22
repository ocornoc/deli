use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deli::{Triangle, Tetrahedron, Degenerable, Collidable};
use nalgebra::Point4;

fn triangle_degeneracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle degeneracy");
    group.sample_size(100);
    group.bench_function("General f32", |b| b.iter(||
        black_box(Triangle::<f32>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).is_degenerate()
    ));
    group.bench_function("General f64", |b| b.iter(||
        black_box(Triangle::<f64>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).is_degenerate()
    ));
    group.bench_function("Point-triangle f32", |b| b.iter(||
        black_box(Triangle::<f32>::new(
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
        )).is_degenerate()
    ));
    group.bench_function("Point-triangle f64", |b| b.iter(||
        black_box(Triangle::<f64>::new(
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
        )).is_degenerate()
    ));
    group.finish();
}

fn tetrahedron_degeneracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tetrahedron degeneracy");
    group.sample_size(100);
    group.bench_function("General f32", |b| b.iter(||
        black_box(Tetrahedron::<f32>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
                Point4::new(10.0, -6.1, 32.9, 4.1),
        )).is_degenerate()
    ));
    group.bench_function("General f64", |b| b.iter(||
        black_box(Tetrahedron::<f64>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
                Point4::new(10.0, -6.1, 32.9, 4.1),
        )).is_degenerate()
    ));
    group.bench_function("Point-tetrahedron f32", |b| b.iter(||
        black_box(Tetrahedron::<f32>::new(
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
        )).is_degenerate()
    ));
    group.bench_function("Point-tetrahedron f64", |b| b.iter(||
        black_box(Tetrahedron::<f64>::new(
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
                Point4::new(0.0, 0.0, 0.0, 0.0),
        )).is_degenerate()
    ));
    group.finish();
}

fn point_triangle_intersect(c: &mut Criterion) {
    let mut group = c.benchmark_group("Point-triangle intersection");
    group.sample_size(100);
    group.bench_function("Noncoplanar f32", |b| b.iter(||
        black_box(Triangle::<f32>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).intersects(
            black_box(&Point4::new(1.0, 3.83333, -1.33333, -2.7))
        )
    ));
    group.bench_function("Coplanar outside f32", |b| b.iter(||
        black_box(Triangle::<f32>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).intersects(
            black_box(&Point4::new(-3.5, -3.5, 5.0, 2.0))
        )
    ));
    group.bench_function("Centroid f32", |b| b.iter(||
        black_box(Triangle::<f32>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).intersects(
            black_box(&Point4::new(0.466667, 3.83333, -1.33333, -2.7))
        )
    ));
    group.bench_function("Noncoplanar f64", |b| b.iter(||
        black_box(Triangle::<f64>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).intersects(
            black_box(&Point4::new(1.0, 3.83333, -1.33333, -2.7))
        )
    ));
    group.bench_function("Coplanar outside f64", |b| b.iter(||
        black_box(Triangle::<f64>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).intersects(
            black_box(&Point4::new(-3.5, -3.5, 5.0, 2.0))
        )
    ));
    group.bench_function("Centroid f64", |b| b.iter(||
        black_box(Triangle::<f64>::new(
                Point4::new(3.5, 5.5, -7.0, -2.0),
                Point4::new(0.0, 1.0, -1.0, 0.0),
                Point4::new(-2.1, 5.0, 4.0, -6.1),
        )).intersects(
            black_box(&Point4::new(0.466667, 3.83333, -1.33333, -2.7))
        )
    ));
    group.finish();
}

criterion_group!(
    microbenches,
    triangle_degeneracy,
    tetrahedron_degeneracy,
    point_triangle_intersect);
criterion_main!(microbenches);