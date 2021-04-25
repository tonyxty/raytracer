use nalgebra::{ArrayStorage, Vector3};
use rand::Rng;
use rand_distr::{Distribution, UnitSphere};
use rand_distr::num_traits::Pow;

use crate::object::Intersection;
use crate::ray::Ray;
use crate::RNG;

pub trait Material {
    fn scatter(&self, int: &Intersection) -> (Ray<f64>, Vector3<f64>);
}

pub struct Metal {
    color: Vector3<f64>,
    fuzz: f64,
}

impl Metal {
    pub fn new(color: Vector3<f64>, fuzz: f64) -> Self {
        Self { color, fuzz }
    }
}

impl Material for Metal {
    fn scatter(&self, int: &Intersection) -> (Ray<f64>, Vector3<f64>) {
        let v = int.ray().direction();
        let n = int.normal();
        let r = reflect(v, n) + self.fuzz * random_unit_vector();
        (Ray::new(*int.point(), r), self.color)
    }
}

pub struct Lambertian {
    color: Vector3<f64>,
}

impl Lambertian {
    pub fn new(color: Vector3<f64>) -> Self {
        Self { color }
    }
}

impl Material for Lambertian {
    fn scatter(&self, int: &Intersection) -> (Ray<f64>, Vector3<f64>) {
        (Ray::new(*int.point(), int.normal() + random_unit_vector()), self.color)
    }
}

pub struct Dielectric {
    index_refraction: f64,
}

impl Dielectric {
    pub fn new(index_refraction: f64) -> Self {
        Self { index_refraction }
    }
}

impl Material for Dielectric {
    fn scatter(&self, int: &Intersection) -> (Ray<f64>, Vector3<f64>) {
        let ratio = if int.front() { 1.0 / self.index_refraction } else { self.index_refraction };
        let v = int.ray().direction();
        let n = int.normal();
        (Ray::new(*int.point(), refract_schlick(v, n, ratio)), Vector3::new(1.0, 1.0, 1.0))
    }
}

fn random_unit_vector() -> Vector3<f64> {
    Vector3::from_data(ArrayStorage([RNG.with(|r| UnitSphere.sample(&mut *r.borrow_mut()))]))
}

fn reflect(v: &Vector3<f64>, n: &Vector3<f64>) -> Vector3<f64> {
    v - 2.0 * v.dot(n) * n
}

fn refract_schlick(v: &Vector3<f64>, n: &Vector3<f64>, ratio: f64) -> Vector3<f64> {
    let c = -v.dot(n).min(1.0);
    let s = (1.0 - c * c).sqrt();
    if ratio * s > 1.0 || reflectance(c, ratio) > RNG.with(|r| r.borrow_mut().gen()) {
        reflect(v, n)
    } else {
        let orthogonal = ratio * (v + c * n);
        let parallel = -(1.0 - orthogonal.norm_squared()).abs().sqrt() * n;
        orthogonal + parallel
    }
}

fn reflectance(c: f64, ratio: f64) -> f64 {
    let r0 = (1.0 - ratio) / (1.0 + ratio);
    let r1 = r0 * r0;
    r1 + (1.0 - r1) * (1.0 - c).pow(5)
}
