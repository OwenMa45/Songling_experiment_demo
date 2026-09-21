"""Compose upstream PiPER assets without editing the submodule."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def vec(values):
    return " ".join(str(float(x)) for x in values)


def build(cfg, destination=None):
    s = cfg["simulation"]
    source = Path(cfg["paths"]["piper_root"]) / "src/piper_description/mujoco_model/piper_description.xml"
    upstream = ET.parse(source).getroot()
    root = ET.Element("mujoco", model="dual_piper_titration")
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep=s["timestep"], integrator="implicitfast", gravity="0 0 -9.81")
    visual = add(root, "visual")
    add(visual, "global", offwidth=s["image_size"], offheight=s["image_size"])
    assets = add(root, "asset")
    world = add(root, "worldbody")
    add(world, "light", pos="0.2 0 1.8", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    add(world, "geom", name="table", type="box", pos="0.25 0 -0.035", size="0.65 0.65 0.035", rgba="0.2 0.25 0.28 1")
    add(world, "camera", name="exterior", pos="1.05 -1.05 0.85", xyaxes="0.707 0.707 0 -0.35 0.35 0.87")
    actuators = add(root, "actuator")
    for arm in ("holder", "squeezer"):
        prefix = arm + "_"
        for element in upstream.find("asset"):
            mesh = copy.deepcopy(element)
            mesh.set("name", prefix + mesh.get("name"))
            mesh.set("file", str((source.parent / mesh.get("file")).resolve()))
            assets.append(mesh)
        base = add(world, "body", name=arm + "_base", pos=vec(s[arm + "_base"]))
        for element in upstream.find("worldbody"):
            node = copy.deepcopy(element)
            for child in node.iter():
                for attr in ("name", "mesh", "joint"):
                    if child.get(attr):
                        child.set(attr, prefix + child.get(attr))
                if child.tag == "joint":
                    child.set("damping", "4" if child.get("type") != "slide" else "10")
                if child.tag == "geom":
                    child.set("name", child.get("mesh") + "_geom")
            base.append(node)
        tool = base.find(f".//body[@name='{prefix}link6']")
        add(tool, "site", name=prefix + "tcp", pos="0 0 0.18", size="0.004")
        if arm == "holder":
            # Local X points down at the scripted grasp pose; tube held across jaws.
            add(tool, "geom", name="glass", type="capsule", fromto="-0.035 0 0.18 0.075 0 0.18", size="0.004", mass="0.015", rgba="0.6 0.85 0.95 0.65")
            add(tool, "geom", name="bulb", type="ellipsoid", pos="-0.055 0 0.18", size="0.022 0.012 0.012", mass="0.008", rgba="0.8 0.18 0.12 1", contype="0", conaffinity="0")
            add(tool, "site", name="tip", pos="0.075 0 0.18", size="0.002")
            add(tool, "site", name="bulb_center", pos="-0.055 0 0.18", size="0.003")
        else:
            add(tool, "camera", name="wrist", pos="0.06 0 0.06", xyaxes="0 1 0 1 0 0", fovy="75")
        for joint in base.iter("joint"):
            slide = joint.get("type") == "slide"
            add(actuators, "position", name=joint.get("name"), joint=joint.get("name"), kp="400" if slide else "700", kv="20" if slide else "40", ctrlrange=joint.get("range"), forcerange="-30 30" if slide else "-100 100")
    # Segmented tube wall (a solid cylinder would incorrectly fill the opening).
    tube = np.array(s["tube_center"])
    for i in range(24):
        a = i * 2 * np.pi / 24
        pos = tube + [s["tube_radius"] * np.cos(a), s["tube_radius"] * np.sin(a), -0.055]
        add(world, "geom", name=f"tube_wall_{i}", type="capsule", pos=vec(pos), size="0.0015 0.055", rgba="0.65 0.9 1 0.4")
    add(world, "geom", name="tube_bottom", type="cylinder", pos=vec(tube - [0, 0, 0.11]), size=f'{s["tube_radius"]} 0.002', rgba="0.65 0.9 1 0.4")
    # Permanently mounted holder is a documented idealization. Exclude its own tool contacts.
    contact = add(root, "contact")
    for arm in ("holder", "squeezer"):
        add(contact, "exclude", body1=arm + "_link7", body2=arm + "_link8")
    xml = ET.tostring(root, encoding="unicode")
    if destination:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(xml, encoding="utf-8")
    return xml
