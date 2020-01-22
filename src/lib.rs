//pub mod bounding;
pub mod collision;
pub mod geometry;
pub use geometry::{
    Triangle, Tetrahedron, Pentachoron,
    Measure, Degenerable, Decomposable
};
pub use collision::{
    Collidable, CollisionDescribable
};