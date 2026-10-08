"""Optional branch-local sole contacts fitted to the existing MOS9 visuals."""
from pathlib import Path
import xml.etree.ElementTree as ET


def model_path(flat_soles=False):
    source = Path("assets/MOS9/MOS9_walk.urdf").resolve()
    if not flat_soles:
        return source
    root = ET.parse(source).getroot()
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((source.parent/mesh.get("filename")).resolve()))
    for name, sign in (("Rfoot", 1), ("Lfoot", -1)):
        link = next(link for link in root.findall("link") if link.get("name") == name)
        for collision in link.findall("collision"):
            link.remove(collision)
        collision = ET.SubElement(link, "collision")
        # Visual-mesh world bounds at zero pose: x [-.072,.078],
        # y [-.045,.045], z [-.059,.0275]. Fit the bottom 10 mm plate.
        ET.SubElement(collision, "origin", xyz=f"0 -0.054 {sign*0.003}", rpy="0 0 0")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="0.09 0.01 0.15")
    output = Path("runs/mos9_teacher/MOS9_flat_soles.urdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output.resolve()
