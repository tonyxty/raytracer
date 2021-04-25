use std::ops::Range;

use lazycell::LazyCell;
use nalgebra::Vector3;

use crate::geometry::Geometry;
use crate::material::Material;
use crate::ray::Ray;

#[derive(Default)]
struct Cache {
    point: LazyCell<Vector3<f64>>,
    normal_front: LazyCell<(Vector3<f64>, bool)>,
}

pub struct Intersection<'g> {
    t: f64,
    ray: Ray<f64>,
    object: &'g dyn Object,
    cache: Cache,
}

impl Intersection<'_> {
    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn ray(&self) -> &Ray<f64> {
        &self.ray
    }

    pub fn point(&self) -> &Vector3<f64> {
        self.cache.point.borrow_with(|| self.ray.at(self.t))
    }

    fn normal_front(&self) -> &(Vector3<f64>, bool) {
        self.cache.normal_front.borrow_with(|| {
            let n = self.object.normal(self.point());
            let front = self.ray.direction().dot(&n) < 0.0;
            (if front { n } else { -n }, front)
        })
    }

    pub fn normal(&self) -> &Vector3<f64> {
        &self.normal_front().0
    }

    pub fn front(&self) -> bool {
        self.normal_front().1
    }

    pub fn scatter(&self) -> (Ray<f64>, Vector3<f64>) {
        self.object.scatter(self)
    }
}

pub trait Object {
    fn intersect(&self, ray: &Ray<f64>, range: Range<f64>) -> Option<Intersection>;
    fn normal(&self, point: &Vector3<f64>) -> Vector3<f64>;
    fn scatter(&self, int: &Intersection) -> (Ray<f64>, Vector3<f64>);
}

impl<G: Geometry, M: Material> Object for (G, M) {
    fn intersect(&self, ray: &Ray<f64>, range: Range<f64>) -> Option<Intersection> {
        self.0.intersect(ray, range).map(|t| Intersection {
            t,
            ray: ray.clone(),
            object: self,
            cache: Default::default(),
        })
    }

    fn normal(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.0.normal(point)
    }

    fn scatter(&self, int: &Intersection) -> (Ray<f64>, Vector3<f64>) {
        self.1.scatter(int)
    }
}
