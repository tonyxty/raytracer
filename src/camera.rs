use nalgebra::Vector3;
use rand_distr::{Distribution, UnitDisc};

use crate::ray::Ray;
use crate::RNG;

pub struct Camera {
    horizontal: Vector3<f64>,
    vertical: Vector3<f64>,
    origin: Vector3<f64>,
    direction: Vector3<f64>,
    right: Vector3<f64>,
    up: Vector3<f64>,
    lens_radius: f64,
}

impl Camera {
    pub fn look_at(
        origin: Vector3<f64>,
        at: &Vector3<f64>,
        up: &Vector3<f64>,
        fov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_distance: f64,
    ) -> Self {
        let viewport_height = 2.0 * (fov / 2.0).tan();
        let focus_plane_height = viewport_height * focus_distance;
        let focus_plane_width = focus_plane_height * aspect_ratio;

        let front = (at - origin).normalize();
        let right = front.cross(up).normalize();
        let up = right.cross(&front);

        let horizontal = right * focus_plane_width;
        let vertical = up * focus_plane_height;
        let direction = front * focus_distance;
        Self {
            horizontal,
            vertical,
            origin,
            direction,
            right,
            up,
            lens_radius: aperture / 2.0,
        }
    }

    pub fn ray_at(&self, u: f64, v: f64) -> Ray<f64> {
        let [x, y]: [f64; 2] = RNG.with(|r| UnitDisc.sample(&mut *r.borrow_mut()));
        let offset = self.lens_radius * (self.right * x + self.up * y);
        let direction = self.direction + self.horizontal * (u - 0.5) + self.vertical * (v - 0.5);
        Ray::new(self.origin + offset, (direction - offset).normalize())
    }
}
