"""Check the grasp against the native finger geometry rather than only its commanded TCP."""

import importlib.util
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2] / "examples" / "coupled_high_accuracy"


def load_demo_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "demos/w1_courier_transport" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 1)])
def test_closed_native_fingers_clear_payload_and_table(side, index):
    motion = load_demo_module("courier_motion_geometry_test", "w1_motion.py")
    geometry = load_demo_module("courier_payload_geometry_test", "../piper_package_transport/geometry.py")
    robot = ET.parse(motion.ASSET).getroot()
    points, opening, quaternion = motion.target(motion.CLOSE_END)
    rotation = Rotation.from_quat(quaternion)
    payload_half_extent = np.abs(motion.ROTATIONS[index]) @ (np.asarray(geometry.PAYLOAD_SIZE) / 2)
    payload_near_x = motion.CENTERS[index, 0] - payload_half_extent[0]
    for finger in (1, 2):
        link = robot.find(f"link[@name='{side}_finger{finger}']")
        mesh = motion.ASSET.parent / link.find("collision/geometry/mesh").get("filename")
        vertices = np.array(
            [list(map(float, line.split()[1:4])) for line in mesh.read_text().splitlines() if line.startswith("v ")]
        )
        pad = np.asarray(list(product((-1, 1), repeat=3))) * motion.FINGER_PAD_HALF_EXTENTS
        pad += motion.FINGER_PAD_CENTER
        vertices = np.vstack((vertices, pad))
        joint = robot.find(f"joint[@name='{side.upper()}_FINGER{finger}_JOINT']")
        origin = joint.find("origin")
        local_rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" "))
        local = local_rotation.apply(vertices) + np.fromstring(origin.get("xyz"), sep=" ")
        world = rotation.apply(local - np.asarray(motion.TCP)) + points[index]
        # Include the carton's 6 mm contact margin and the film's 1.5 mm contact radius.
        assert world[:, 0].max() + 0.006 + 0.0015 < payload_near_x
        assert world[:, 2].min() > motion.TABLE_Z + 0.0015
    assert opening == 0


def test_closed_grip_has_a_stationary_dwell_before_lifting():
    motion = load_demo_module("courier_motion_dwell_test", "w1_motion.py")
    assert motion.LIFT_START - motion.CLOSE_END >= 0.5
    for t in np.linspace(motion.CLOSE_END, motion.LIFT_START, 31):
        points, opening, _ = motion.target(t)
        np.testing.assert_allclose(points, motion.GRASPS)
        assert opening == 0
    assert np.all(motion.target(6)[0][:, 2] > motion.GRASPS[:, 2])
    np.testing.assert_allclose(motion.target(7)[0][:, 0], 0.410)
    assert motion.target(10)[1] == 0
    assert motion.target(10.2)[1] == pytest.approx(0.045)


def test_rubber_pads_preserve_mass_and_robot_collision_filters():
    import newton

    motion = load_demo_module("courier_pad_filter_test", "w1_motion.py")
    builder = newton.ModelBuilder()
    world = builder.add_shape_box(body=-1, hx=0.2, hy=0.2, hz=0.01)
    first_robot_shape = builder.shape_count
    bodies = []
    for side, finger in product(("left", "right"), (1, 2)):
        body = builder.add_body(label=f"robot/{side}_finger{finger}")
        builder.add_shape_box(body=body, hx=0.01, hy=0.03, hz=0.01)
        bodies.append(body)
    mass = np.asarray(builder.body_mass).copy()
    first_pad = builder.shape_count
    motion.add_finger_pads(builder, first_robot_shape)
    np.testing.assert_array_equal(builder.body_mass, mass)
    assert builder.shape_count == first_pad + 4
    assert builder.shape_body[first_pad:] == bodies
    filters = set(builder.shape_collision_filter_pairs)
    for pad in range(first_pad, builder.shape_count):
        assert (world, pad) not in filters
        assert builder.shape_flags[pad] & int(newton.ShapeFlags.COLLIDE_PARTICLES)
        assert np.allclose(np.asarray(builder.shape_scale[pad]), motion.FINGER_PAD_HALF_EXTENTS)
        for other in range(first_robot_shape, pad):
            assert (other, pad) in filters


def test_grasp_observations_detect_an_open_hand_and_a_dropped_bag():
    validation = load_demo_module("courier_grasp_validation_test", "validate_w1.py")
    fingers = np.zeros((2, 2, 7))
    fingers[:, :, 2] = 1.2
    fingers[:, 1, 1] = 0.01
    film = np.array([[[0, 0.005, 1.18]], [[0, 0.005, 1.18]]])
    held = validation.grasp_observations(film, fingers)
    assert held["maximum_jaw_gap_m"] < 0.030
    assert held["maximum_film_finger_distance_m"] < 0.080
    fingers[1, 1, 1] = 0.09
    assert validation.grasp_observations(film, fingers)["maximum_jaw_gap_m"] > 0.030
    film[1, :, 2] = 0.88
    assert validation.grasp_observations(film, fingers)["maximum_film_finger_distance_m"] > 0.080
