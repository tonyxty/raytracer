use nalgebra::{ClosedAdd, ClosedMul, Scalar, Vector3};

#[derive(Clone)]
pub struct Ray<T> {
    pub origin: Vector3<T>,
    direction: Vector3<T>,
}

impl<T> Ray<T> {
    pub const fn new(origin: Vector3<T>, direction: Vector3<T>) -> Self {
        Self { origin, direction }
    }
}

impl<T: Scalar + ClosedAdd + ClosedMul> Ray<T> {
    pub fn at(&self, t: T) -> Vector3<T> {
        &self.origin + &self.direction * t
    }

    pub fn direction(&self) -> &Vector3<T> {
        &self.direction
    }
}
