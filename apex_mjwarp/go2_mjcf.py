"""Generate a collision-only MuJoCo Go2 model from APEX's Go2 URDF.

The source URDF remains authoritative for masses, inertias, joint axes,
limits, and primitive collision geometry. Visual DAE meshes are omitted.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


def _values(text: str | None, default: str) -> list[float]:
    return [float(x) for x in (text or default).split()]


def _quat(rpy: list[float]) -> str:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
    return " ".join(f"{x:.10g}" for x in q)


def _pose(node: ET.Element | None) -> dict[str, str]:
    if node is None:
        return {}
    xyz = _values(node.get("xyz"), "0 0 0")
    rpy = _values(node.get("rpy"), "0 0 0")
    return {"pos": " ".join(str(x) for x in xyz), "quat": _quat(rpy)}


def build_go2_mjcf(urdf_path: str | Path) -> str:
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    children: dict[str, list[ET.Element]] = {name: [] for name in links}
    child_names = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children[parent].append(joint)
        child_names.add(child)
    roots = set(links) - child_names
    if roots != {"base"}:
        raise ValueError(f"Expected one Go2 base link; got {roots}")

    model = ET.Element("mujoco", {"model": "APEX_Go2"})
    ET.SubElement(model, "compiler", {"angle": "radian", "inertiafromgeom": "false"})
    ET.SubElement(model, "option", {"timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "Euler"})
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(world, "geom", {"name": "ground", "type": "plane", "size": "10 10 0.1", "friction": "1 0.005 0.0001"})
    actuators = ET.SubElement(model, "actuator")

    def add_link(name: str, parent: ET.Element, connecting_joint: ET.Element | None = None) -> None:
        attrs = {"name": name}
        if connecting_joint is not None:
            attrs.update(_pose(connecting_joint.find("origin")))
        body = ET.SubElement(parent, "body", attrs)
        if connecting_joint is None:
            ET.SubElement(body, "freejoint", {"name": "root"})
        elif connecting_joint.get("type") != "fixed":
            limit = connecting_joint.find("limit")
            jattrs = {"name": connecting_joint.get("name"), "type": "hinge",
                      "axis": connecting_joint.find("axis").get("xyz"),
                      "range": f'{limit.get("lower")} {limit.get("upper")}', "limited": "true"}
            ET.SubElement(body, "joint", jattrs)
            ET.SubElement(actuators, "motor", {"name": connecting_joint.get("name"),
                                               "joint": connecting_joint.get("name"), "gear": "1"})

        link = links[name]
        inertial = link.find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 0:
            inertia = inertial.find("inertia")
            iattrs = {"mass": inertial.find("mass").get("value"),
                      "fullinertia": " ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))}
            # MuJoCo's fullinertia defines the inertia orientation itself.
            # The APEX Go2 URDF has identity inertial rotations throughout.
            origin = inertial.find("origin")
            if any(abs(x) > 1e-10 for x in _values(origin.get("rpy"), "0 0 0")):
                raise ValueError(f"Non-identity inertial rotation in {name}")
            iattrs["pos"] = _pose(origin)["pos"]
            ET.SubElement(body, "inertial", iattrs)

        for idx, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            shape = next(iter(geometry))
            geom = {"name": f"{name}_collision_{idx}", "friction": "1 0.005 0.0001"}
            geom.update(_pose(collision.find("origin")))
            if shape.tag == "box":
                geom.update(type="box", size=" ".join(str(x / 2) for x in _values(shape.get("size"), "0 0 0")))
            elif shape.tag == "sphere":
                geom.update(type="sphere", size=shape.get("radius"))
            elif shape.tag == "cylinder":
                # Isaac Gym's APEX asset option replaces cylinders with capsules.
                geom.update(type="capsule", size=f'{shape.get("radius")} {float(shape.get("length")) / 2}')
            else:
                raise ValueError(f"Unsupported collision geometry: {shape.tag} in {name}")
            ET.SubElement(body, "geom", geom)
        for joint in children[name]:
            add_link(joint.find("child").get("link"), body, joint)

    add_link("base", world)
    return ET.tostring(model, encoding="unicode")


def write_go2_mjcf(urdf_path: str | Path, target: str | Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_go2_mjcf(urdf_path), encoding="utf-8")
    return target
