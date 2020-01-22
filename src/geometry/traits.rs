use nalgebra::ComplexField;

/// Implements the square of the measure.
pub trait Measure {
    type Output: ComplexField;
    
    /// Returns the square of the measure.
    fn measure_squared(&self) -> Self::Output;
    
    /// Returns the measure of the object.
    #[inline]
    fn measure(&self) -> Self::Output {
        self.measure_squared().sqrt()
    }
}

/// A trait describing whether a type can be degenerate. What "degenerate"
/// actually means is implementation defined and should be documented well.
pub trait Degenerable {
    /// Returns whether the object is 'degenerate'.
    fn is_degenerate(&self) -> bool;
}

/// A trait describing a type that can be completely decomposed into a 
/// collection of another type.
pub trait Decomposable<T>: Sized {
    /// The output type.
    type Output;
    /// Decomposes an object into its pieces.
    fn decompose(&self) -> Self::Output;
    /// Reconstructs an object from its pieces.
    fn recompose(components: &Self::Output) -> Option<Self>;
}

/// Describes objects that can transform a cartesian point into an areal point.
pub trait ArealCoordinates {
    /// The output type. Holds the output points.
    type Output;
    /// The type of the input points.
    type PointType;
    /// Turns Cartesian coordinates into Areal coordinates.
    fn get_areal_of_cart(&self, p: &Self::PointType) -> Self::Output;
}