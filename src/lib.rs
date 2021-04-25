#![feature(box_syntax)]

use std::borrow::Borrow;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::ops::Range;

use itertools::iproduct;
use nalgebra::Vector3;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;

use crate::camera::Camera;
use crate::geometry::Sphere;
use crate::material::{Dielectric, Lambertian, Metal};
use crate::object::Object;
use crate::ray::Ray;

mod camera;
mod geometry;
mod material;
mod object;
mod ray;

const NUM_SAMPLES: u32 = 128;
const NUM_THREADS: u32 = 8;
const RANDOM_RANGE: Range<i32> = -11..11;
const IMAGE_WIDTH: u32 = 300;
const IMAGE_HEIGHT: u32 = 200;

thread_local! {
    pub(crate) static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_rng(rand::thread_rng()).unwrap());
}

fn ray_color<R: Borrow<dyn Object + Sync>>(objects: &[R], ray: &Ray<f64>, depth: usize) -> Vector3<f64> {
    if depth > 0 {
        objects.iter()
            .filter_map(|o| o.borrow().intersect(ray, 0.0..f64::INFINITY))
            .min_by(|x, y| x.t().partial_cmp(&y.t()).expect("some compare thing failed"))
            .map(|i| {
                let (r, m) = i.scatter();
                ray_color(objects, &r, depth - 1).component_mul(&m)
            })
            .unwrap_or_else(|| {
                let v = ray.direction();
                let t = 0.5 * (v.y + 1.0);
                Vector3::new(1.0 - t, 1.0 - t, 1.0 - t) + t * Vector3::new(0.5, 0.7, 1.0)
            })
    } else { Default::default() }
}

fn worker<R: Borrow<dyn Object + Sync>>(
    camera: &Camera, objects: &[R],
    width: u32, height: u32, i: u32, j: u32,
) -> Vector3<f64> {
    (0..NUM_SAMPLES).map(|_| {
        let u = (i as f64 + RNG.with(|r| r.borrow_mut().gen_range(-0.5..0.5))) / (width as f64);
        let v = 1.0 - (j as f64 + RNG.with(|r| r.borrow_mut().gen_range(-0.5..0.5))) / (height as f64);
        let ray = camera.ray_at(u, v);
        ray_color(objects, &ray, 20)
    }).sum::<Vector3<f64>>() / (NUM_SAMPLES as f64)
}

fn create_camera() -> Camera {
    Camera::look_at(
        Vector3::new(13.0, 2.0, 3.0),
        &Vector3::new(0.0, 0.0, 0.0),
        &Vector3::new(0.0, 1.0, 0.0),
        PI / 9.0,
        3.0 / 2.0,
        0.1,
        10.0,
    )
}

fn random_range(range: Range<f64>) -> f64 {
    RNG.with(|r| r.borrow_mut().gen_range(range))
}

fn random_vector(range: Range<f64>) -> Vector3<f64> {
    let uniform = Uniform::from(range);
    let x = RNG.with(|r| uniform.sample(&mut *r.borrow_mut()));
    let y = RNG.with(|r| uniform.sample(&mut *r.borrow_mut()));
    let z = RNG.with(|r| uniform.sample(&mut *r.borrow_mut()));
    Vector3::new(x, y, z)
}

fn create_scene() -> Vec<Box<dyn Object + Sync>> {
    let mut scene = iproduct!(RANDOM_RANGE, RANDOM_RANGE)
        .filter_map(|(a, b)| -> Option<Box<dyn Object + Sync>> {
            let x = a as f64 + random_range(0.0..0.9);
            let y = 0.2;
            let z = b as f64 + random_range(0.0..0.9);
            let center = Vector3::new(x, y, z);
            if (center - Vector3::new(4.0, 0.2, 0.0)).norm_squared() > 0.81 {
                let sphere = Sphere::new(center, 0.2);
                let choose_material = RNG.with(|r| r.borrow_mut().gen::<f64>());
                Some(if choose_material < 0.8 {
                    let color = random_vector(0.0..1.0).component_mul(&random_vector(0.0..1.0));
                    box (sphere, Lambertian::new(color))
                } else if choose_material < 0.95 {
                    let color = random_vector(0.5..1.0);
                    let fuzz = random_range(0.0..0.5);
                    box (sphere, Metal::new(color, fuzz))
                } else {
                    box (sphere, Dielectric::new(1.5))
                })
            } else { None }
        }).collect::<Vec<_>>();
    scene.push(box (
        Sphere::new(Vector3::new(0.0, -1000.0, 0.0), 1000.0),
        Lambertian::new(Vector3::new(0.5, 0.5, 0.5))
    ));
    scene.push(box (
        Sphere::new(Vector3::new(0.0, 1.0, 0.0), 1.0),
        Dielectric::new(1.5)
    ));
    scene.push(box (
        Sphere::new(Vector3::new(-4.0, 1.0, 0.0), 1.0),
        Lambertian::new(Vector3::new(0.4, 0.2, 0.1))
    ));
    scene.push(box (
        Sphere::new(Vector3::new(4.0, 1.0, 0.0), 1.0),
        Metal::new(Vector3::new(0.7, 0.6, 0.5), 0.0)
    ));
    scene
}

pub fn render() -> (u32, u32, Vec<Vector3<f64>>) {
    let camera = create_camera();
    let scene = create_scene();
    let objects = &scene[..];

    let mut results = crossbeam::scope(|s| {
        let threads = (0..NUM_THREADS).map(|_| {
            s.spawn(|_| {
                iproduct!(0..IMAGE_WIDTH, 0..IMAGE_HEIGHT)
                    .map(|(i, j)| worker(&camera, objects, IMAGE_WIDTH, IMAGE_HEIGHT, i, j))
                    .collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>();
        threads.into_iter().map(|t| t.join().unwrap()).collect::<Vec<_>>()
    }).unwrap().into_iter();

    let mut buffer = results.next().unwrap();
    results.for_each(|r| {
        buffer.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
    });
    buffer.iter_mut().for_each(|x| {
        *x = (*x / NUM_THREADS as f64).map(f64::sqrt);
    });
    (IMAGE_WIDTH, IMAGE_HEIGHT, buffer)
}

pub fn write_to_file(path: &str, image: (u32, u32, Vec<Vector3<f64>>)) {
    let mut file = File::create(path).unwrap();
    let (width, height, buffer) = image;
    writeln!(file, "{} {}", width, height).unwrap();
    buffer.iter().for_each(|c| {
        let color = c.map(|x| (x * 255.0) as u8);
        writeln!(file, "{} {} {}", color.x, color.y, color.z).unwrap();
    });
}

pub fn read_from_file(path: &str) -> (u32, u32, Vec<Vector3<f64>>) {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let header = lines.next().unwrap().unwrap();
    let (width, height) = {
        let mut i = header
            .split_ascii_whitespace()
            .map(|s| s.parse::<u32>().unwrap());
        (i.next().unwrap(), i.next().unwrap())
    };
    let buffer = lines.map(|s| Vector3::from_iterator(
        s.unwrap()
            .split_ascii_whitespace()
            .map(|s| s.parse::<u32>().unwrap() as f64 / 255.0)
    )).collect();
    (width, height, buffer)
}

#[cfg(feature = "sdl2")]
pub fn show_image(image: (u32, u32, Vec<Vector3<f64>>)) {
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;
    use sdl2::pixels::Color;
    use sdl2::rect::Point;

    let (width, height, buffer) = image;
    let sdl = sdl2::init().unwrap();
    let window_subsystem = sdl.video().unwrap();
    let window = window_subsystem
        .window("Raytracer", width, height)
        .build()
        .unwrap();
    let mut canvas = window.into_canvas().build().unwrap();
    iproduct!(0..width, 0..height)
        .zip(buffer.into_iter())
        .for_each(|((i, j), c)| {
            let color = c.map(|x| (x * 255.0) as u8);
            canvas.set_draw_color(Color::RGB(color.x, color.y, color.z));
            canvas.draw_point(Point::new(i as i32, j as i32)).unwrap();
        });
    canvas.present();
    let mut event_pump = sdl.event_pump().unwrap();
    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    return;
                }
                _ => {}
            }
        }
    }
}
