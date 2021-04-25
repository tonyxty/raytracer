use std::ops::Range;

use nalgebra::Vector3;

use crate::ray::Ray;

pub trait Geometry {
    fn intersect(&self, ray: &Ray<f64>, range: Range<f64>) -> Option<f64>;
    fn normal(&self, point: &Vector3<f64>) -> Vector3<f64>;
}

pub struct Sphere {
    center: Vector3<f64>,
    radius: f64,
}

impl Sphere {
    pub const fn new(center: Vector3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Geometry for Sphere {
    fn intersect(&self, ray: &Ray<f64>, range: Range<f64>) -> Option<f64> {
        let v = ray.origin - self.center;
        let a = ray.direction().norm_squared();
        let b = ray.direction().dot(&v);
        let c = v.norm_squared() - self.radius * self.radius;
        let disc = b * b - a * c;
        if disc > 0.0 {
            let d = disc.sqrt();
            [(-b - d) / a, (-b + d) / a].iter().copied()
                .find(|t| range.contains(t))
        } else {
            None
        }
    }

    fn normal(&self, point: &Vector3<f64>) -> Vector3<f64> {
        (point - self.center).normalize()
    }
}
