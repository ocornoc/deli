pub trait Collidable<Other: ?Sized> {
    /// Returns whether the `Other` is at least partially contained in `self`.
    /// This does include the degenerate case of two boundaries touching, if
    /// the concept of boundaries applies. Should be reflexive and is assumed
    /// so by the library.
    fn intersects(self: &Self, col: &Other) -> bool;
}

pub trait CollisionDescribable<Other: ?Sized> {
    type CollisionType;
    
    /// Returns the type of collision between two objects. Presuming
    /// `CollisionType` isn't `None` but is `Option`, `intersects` should be
    /// true. Should be reflexive and is assumed so by the library.
    fn get_collision(self: &Self, col: &Other) -> Self::CollisionType;
}

/// Automatically implement `Collidable<Other>` for types implementing trait
/// `CollisionDescribable<Option<Other>>`, with the `intersects` function
/// returning false iff the collision is `None`. Intended for consistency
/// between `CollisionDescribable` and `Collidable`.
impl<CT, Other, T> Collidable<Other> for T
where Other: ?Sized,
      T: CollisionDescribable<Other, CollisionType=Option<CT>>
{
    #[inline]
    fn intersects(self: &Self, col: &Other) -> bool {
        self.get_collision(col).is_some()
    }
}