import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from tpu_builder import (
    TPUMembraneBuilder,
    TPUMembraneConfig,
    TPUPermeabilityPredictor,
    MoleculeDescriptor
)

# Page config
st.set_page_config(
    page_title="TPU Membrane Builder",
    layout="wide"
)

st.title("TPU Membrane Builder")

# Initialize session state
if 'membrane' not in st.session_state:
    st.session_state.membrane = None
if 'perm_result' not in st.session_state:
    st.session_state.perm_result = None


def generate_polymer_chains(membrane, sparsa_frac, carbosil_frac, num_chains=25):
    """
    Generate a dense rectangular membrane slab filled with polymer chains.

    Creates a box-shaped cross-section of the membrane with tangled
    polymer chains that fill the volume densely.
    """
    atoms = []
    atom_id = 1
    res_id = 1

    np.random.seed(42)

    # Rectangular membrane slab dimensions (Angstroms)
    # Wide and tall, but thin in Z (membrane cross-section)
    box_x = 40  # width
    box_y = 40  # height
    box_z = 15  # thickness (membrane thickness direction)

    def add_atom(element, x, y, z, res_name):
        nonlocal atom_id, res_id
        # Clamp atoms to stay within the box
        x = max(-box_x/2, min(box_x/2, x))
        y = max(-box_y/2, min(box_y/2, y))
        z = max(-box_z/2, min(box_z/2, z))
        atoms.append({
            'id': atom_id,
            'name': element,
            'res_name': res_name,
            'res_id': res_id,
            'x': x, 'y': y, 'z': z,
            'element': element
        })
        atom_id += 1

    def random_direction():
        """Generate random unit vector, biased to stay in XY plane"""
        theta = np.random.uniform(0, 2 * np.pi)
        # Bias phi to be mostly horizontal (XY plane)
        phi = np.random.uniform(np.pi/3, 2*np.pi/3)
        return (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi) * 0.3  # Reduce Z component
        )

    def step(x, y, z, bond_len, direction, wobble=0.4):
        """Take a step along chain with randomness"""
        dx, dy, dz = direction
        dx += np.random.uniform(-wobble, wobble)
        dy += np.random.uniform(-wobble, wobble)
        dz += np.random.uniform(-wobble * 0.3, wobble * 0.3)  # Less Z wobble
        mag = np.sqrt(dx*dx + dy*dy + dz*dz)
        if mag > 0:
            dx, dy, dz = dx/mag * bond_len, dy/mag * bond_len, dz/mag * bond_len
        new_x, new_y, new_z = x + dx, y + dy, z + dz

        # Bounce off walls
        if abs(new_x) > box_x/2:
            dx = -dx
            new_x = x + dx
        if abs(new_y) > box_y/2:
            dy = -dy
            new_y = y + dy
        if abs(new_z) > box_z/2:
            dz = -dz
            new_z = z + dz

        return new_x, new_y, new_z, (dx/bond_len if bond_len > 0 else 0,
                                      dy/bond_len if bond_len > 0 else 0,
                                      dz/bond_len if bond_len > 0 else 0)

    def generate_peg_segment(x, y, z, direction, n_units=6):
        """PEG/PPG: -[CH2-CH2-O]n- polyether soft segment"""
        nonlocal res_id
        for _ in range(n_units):
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "PEG")
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "PEG")
            x, y, z, direction = step(x, y, z, 1.43, direction)
            add_atom("O", x, y, z, "PEG")
            res_id += 1
        return x, y, z, direction

    def generate_pcl_segment(x, y, z, direction, n_units=4):
        """PCL/Polyester: -[O-CO-(CH2)5]n- soft segment"""
        nonlocal res_id
        for _ in range(n_units):
            x, y, z, direction = step(x, y, z, 1.43, direction)
            add_atom("O", x, y, z, "PCL")
            x, y, z, direction = step(x, y, z, 1.33, direction)
            add_atom("C", x, y, z, "PCL")
            add_atom("O", x + np.random.uniform(-0.6, 0.6), y + 0.8, z, "PCL")
            for _ in range(3):
                x, y, z, direction = step(x, y, z, 1.54, direction)
                add_atom("C", x, y, z, "PCL")
            res_id += 1
        return x, y, z, direction

    def generate_pdms_segment(x, y, z, direction, n_units=5):
        """PDMS: -[Si(CH3)2-O]n- silicone soft segment"""
        nonlocal res_id
        for _ in range(n_units):
            x, y, z, direction = step(x, y, z, 1.64, direction)
            add_atom("SI", x, y, z, "PDM")
            # Methyl groups
            add_atom("C", x + np.random.uniform(0.6, 1.0), y + np.random.uniform(-0.8, 0.8), z, "PDM")
            add_atom("C", x + np.random.uniform(-1.0, -0.6), y + np.random.uniform(-0.8, 0.8), z, "PDM")
            x, y, z, direction = step(x, y, z, 1.64, direction)
            add_atom("O", x, y, z, "PDM")
            res_id += 1
        return x, y, z, direction

    def generate_urethane_hard(x, y, z, direction):
        """H12MDI urethane hard segment"""
        nonlocal res_id
        x, y, z, direction = step(x, y, z, 1.47, direction)
        add_atom("N", x, y, z, "URE")
        x, y, z, direction = step(x, y, z, 1.33, direction)
        add_atom("C", x, y, z, "URE")
        add_atom("O", x + np.random.uniform(-0.4, 0.4), y + 0.9, z, "URE")
        x, y, z, direction = step(x, y, z, 1.43, direction)
        add_atom("O", x, y, z, "URE")
        for _ in range(3):
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "URE")
        x, y, z, direction = step(x, y, z, 1.43, direction)
        add_atom("O", x, y, z, "URE")
        x, y, z, direction = step(x, y, z, 1.33, direction)
        add_atom("C", x, y, z, "URE")
        add_atom("O", x + np.random.uniform(-0.4, 0.4), y + 0.9, z, "URE")
        x, y, z, direction = step(x, y, z, 1.47, direction)
        add_atom("N", x, y, z, "URE")
        res_id += 1
        return x, y, z, direction

    def generate_pc_urethane_hard(x, y, z, direction):
        """Polycarbonate-urethane hard segment for CarboSil"""
        nonlocal res_id
        x, y, z, direction = step(x, y, z, 1.43, direction)
        add_atom("O", x, y, z, "PCB")
        x, y, z, direction = step(x, y, z, 1.33, direction)
        add_atom("C", x, y, z, "PCB")
        add_atom("O", x + np.random.uniform(-0.4, 0.4), y + 0.9, z, "PCB")
        x, y, z, direction = step(x, y, z, 1.43, direction)
        add_atom("O", x, y, z, "PCB")
        for _ in range(3):
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "PCB")
        x, y, z, direction = step(x, y, z, 1.47, direction)
        add_atom("N", x, y, z, "PCB")
        x, y, z, direction = step(x, y, z, 1.33, direction)
        add_atom("C", x, y, z, "PCB")
        add_atom("O", x + np.random.uniform(-0.4, 0.4), y + 0.9, z, "PCB")
        res_id += 1
        return x, y, z, direction

    # Determine composition
    sparsa_total = sparsa_frac
    carbosil_total = carbosil_frac
    total = sparsa_total + carbosil_total
    if total == 0:
        total = 1
        sparsa_total = 0.5
        carbosil_total = 0.5

    # Generate chains starting from a grid to fill the box evenly
    # Use more chains for denser packing
    n_chains = 60
    grid_nx, grid_ny, grid_nz = 5, 5, 3  # Start points grid

    chain_count = 0
    for gx in range(grid_nx):
        for gy in range(grid_ny):
            for gz in range(grid_nz):
                if chain_count >= n_chains:
                    break

                # Start position on grid with jitter
                x = -box_x/2 + (gx + 0.5) * box_x/grid_nx + np.random.uniform(-2, 2)
                y = -box_y/2 + (gy + 0.5) * box_y/grid_ny + np.random.uniform(-2, 2)
                z = -box_z/2 + (gz + 0.5) * box_z/grid_nz + np.random.uniform(-1, 1)

                direction = random_direction()
                is_carbosil = np.random.random() < (carbosil_total / total)

                # Generate 3-5 repeat units per chain
                for _ in range(np.random.randint(3, 6)):
                    if is_carbosil:
                        x, y, z, direction = generate_pdms_segment(x, y, z, direction, n_units=np.random.randint(4, 7))
                        x, y, z, direction = generate_pc_urethane_hard(x, y, z, direction)
                    else:
                        if np.random.random() < 0.7:
                            x, y, z, direction = generate_peg_segment(x, y, z, direction, n_units=np.random.randint(5, 8))
                        else:
                            x, y, z, direction = generate_pcl_segment(x, y, z, direction, n_units=np.random.randint(3, 5))
                        x, y, z, direction = generate_urethane_hard(x, y, z, direction)

                    # Occasionally change direction for more tangling
                    if np.random.random() < 0.3:
                        direction = random_direction()

                chain_count += 1

    return atoms


def render_3dmol_allatom(atoms, carbosil_frac, molecule_info=None, animate=False):
    """Render all-atom style structure using 3Dmol.js"""

    # Build PDB string
    pdb_lines = []
    for atom in atoms:
        # PDB format
        line = f"ATOM  {atom['id']:5d} {atom['name']:4s} {atom['res_name']:3s}  {atom['res_id']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          {atom['element']:>2s}"
        pdb_lines.append(line)
    pdb_lines.append("END")
    pdb_data = "\n".join(pdb_lines)

    # Escape for JavaScript
    pdb_escaped = pdb_data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    # Color scheme based on composition
    # CarboSil: blue tones, Sparsa: warmer tones
    if carbosil_frac > 0.5:
        # More CarboSil - blue/cyan scheme
        color_scheme = """
        viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.12, color: '0x3498db'}, sphere: {scale: 0.2, color: '0x3498db'}});
        viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.1, color: '0x1abc9c'}, sphere: {scale: 0.18, color: '0x1abc9c'}});
        viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.1, color: '0x2ecc71'}, sphere: {scale: 0.18, color: '0x2ecc71'}});
        viewer.setStyle({resn: 'AMO'}, {stick: {radius: 0.08, color: '0x7f8c8d'}, sphere: {scale: 0.15, color: '0x7f8c8d'}});
        """
    else:
        # More Sparsa - orange/red scheme
        color_scheme = """
        viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.12, color: '0xe74c3c'}, sphere: {scale: 0.2, color: '0xe74c3c'}});
        viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.1, color: '0x9b59b6'}, sphere: {scale: 0.18, color: '0x9b59b6'}});
        viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.1, color: '0xf39c12'}, sphere: {scale: 0.18, color: '0xf39c12'}});
        viewer.setStyle({resn: 'AMO'}, {stick: {radius: 0.08, color: '0x7f8c8d'}, sphere: {scale: 0.15, color: '0x7f8c8d'}});
        """

    # Molecule colors
    mol_colors = {
        "phenol": "0xe74c3c",
        "m-cresol": "0x9b59b6",
        "glucose": "0xf39c12",
        "oxygen": "0x3498db"
    }

    mol_sphere_js = ""
    animation_js = ""

    if molecule_info:
        color = mol_colors.get(molecule_info['name'], "0x1abc9c")

        if animate:
            animation_js = f"""
            var sphereId = null;
            var startZ = 25;
            var endZ = -25;
            var duration = 4000;
            var startTime = Date.now();
            var color = {color};

            function animatePermeation() {{
                var elapsed = Date.now() - startTime;
                var progress = elapsed / duration;

                if (progress >= 1) {{
                    if (sphereId !== null) viewer.removeShape(sphereId);
                    sphereId = viewer.addSphere({{
                        center: {{x: 0, y: 0, z: 0}},
                        radius: 2.5,
                        color: color,
                        opacity: 0.95
                    }});
                    viewer.render();
                    return;
                }}

                var z;
                if (progress < 0.3) {{
                    z = startZ - startZ * (progress / 0.3);
                }} else if (progress < 0.7) {{
                    var membraneProgress = (progress - 0.3) / 0.4;
                    z = 0 - (endZ * 0.5) * membraneProgress;
                }} else {{
                    var exitProgress = (progress - 0.7) / 0.3;
                    z = endZ * 0.5 - (endZ * 0.5) * exitProgress;
                }}

                if (sphereId !== null) viewer.removeShape(sphereId);
                sphereId = viewer.addSphere({{
                    center: {{x: 0, y: 0, z: z}},
                    radius: 2.5,
                    color: color,
                    opacity: 0.95
                }});
                viewer.render();
                requestAnimationFrame(animatePermeation);
            }}

            animatePermeation();
            """
        else:
            mol_sphere_js = f"""
            viewer.addSphere({{
                center: {{x: 0, y: 0, z: 0}},
                radius: 2.5,
                color: {color},
                opacity: 0.95
            }});
            """

    # Box dimensions for bounding box
    box_x, box_y, box_z = 40, 40, 15

    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");

        {color_scheme}

        // Draw membrane bounding box
        var hx = {box_x}/2, hy = {box_y}/2, hz = {box_z}/2;
        var boxColor = 0x555555;
        var boxWidth = 1.5;
        viewer.addLine({{start: {{x: -hx, y: -hy, z: -hz}}, end: {{x: hx, y: -hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: -hz}}, end: {{x: hx, y: hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: -hz}}, end: {{x: -hx, y: hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: -hz}}, end: {{x: -hx, y: -hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: -hy, z: hz}}, end: {{x: hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: hz}}, end: {{x: hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: hz}}, end: {{x: -hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: hz}}, end: {{x: -hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: -hy, z: -hz}}, end: {{x: -hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: -hz}}, end: {{x: hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: -hz}}, end: {{x: hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: -hz}}, end: {{x: -hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});

        {mol_sphere_js}

        viewer.zoomTo();
        viewer.zoom(0.5);
        viewer.rotate(20, {{x: 1, y: 0, z: 0}});
        viewer.rotate(-15, {{x: 0, y: 1, z: 0}});
        viewer.render();

        {animation_js}
    </script>
    """
    components.html(html, height=520)


def render_3dmol_allatom_styled(atoms, carbosil_frac, style, molecule_info=None, animate=False):
    """Render all-atom style structure with style options"""

    # Build PDB string
    pdb_lines = []
    for atom in atoms:
        line = f"ATOM  {atom['id']:5d} {atom['name']:4s} {atom['res_name']:3s}  {atom['res_id']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          {atom['element']:>2s}"
        pdb_lines.append(line)
    pdb_lines.append("END")
    pdb_data = "\n".join(pdb_lines)

    pdb_escaped = pdb_data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    # Style-based rendering
    # Segment colors:
    # PEG = Polyether (Sparsa soft) - Green
    # PCL = Polycaprolactone (Sparsa-2 soft) - Teal
    # PDM = PDMS/Silicone (CarboSil soft) - Blue
    # PCB = Polycarbonate (CarboSil hard) - Purple
    # URE = Urethane (hard segment) - Red/Orange

    if style == "stick":
        color_scheme = """
        // Color by segment type with stick+ball style
        viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.12, colorscheme: 'greenCarbon'}});
        viewer.setStyle({resn: 'PCL'}, {stick: {radius: 0.12, colorscheme: 'cyanCarbon'}});
        viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.14, colorscheme: 'blueCarbon'}});
        viewer.setStyle({resn: 'PCB'}, {stick: {radius: 0.12, colorscheme: 'purpleCarbon'}});
        viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.14, colorscheme: 'orangeCarbon'}});
        // Highlight specific elements
        viewer.setStyle({elem: 'N'}, {stick: {radius: 0.14}, sphere: {scale: 0.25, color: '0x3498db'}});
        viewer.setStyle({elem: 'SI'}, {stick: {radius: 0.16}, sphere: {scale: 0.3, color: '0xf1c40f'}});
        """
    elif style == "sphere":
        color_scheme = """
        // Space-filling CPK style
        viewer.setStyle({resn: 'PEG'}, {sphere: {scale: 0.3, colorscheme: 'greenCarbon'}});
        viewer.setStyle({resn: 'PCL'}, {sphere: {scale: 0.3, colorscheme: 'cyanCarbon'}});
        viewer.setStyle({resn: 'PDM'}, {sphere: {scale: 0.32, colorscheme: 'blueCarbon'}});
        viewer.setStyle({resn: 'PCB'}, {sphere: {scale: 0.3, colorscheme: 'purpleCarbon'}});
        viewer.setStyle({resn: 'URE'}, {sphere: {scale: 0.32, colorscheme: 'orangeCarbon'}});
        viewer.setStyle({elem: 'SI'}, {sphere: {scale: 0.4, color: '0xf1c40f'}});
        """
    else:  # line - color by residue type to show polymer segments clearly
        color_scheme = """
        // Wire frame colored by segment type
        viewer.setStyle({resn: 'PEG'}, {line: {linewidth: 2.5, color: '0x27ae60'}});  // PEG - Green
        viewer.setStyle({resn: 'PCL'}, {line: {linewidth: 2.5, color: '0x16a085'}});  // PCL - Teal
        viewer.setStyle({resn: 'PDM'}, {line: {linewidth: 3, color: '0x2980b9'}});    // PDMS - Blue
        viewer.setStyle({resn: 'PCB'}, {line: {linewidth: 2.5, color: '0x8e44ad'}});  // Polycarbonate - Purple
        viewer.setStyle({resn: 'URE'}, {line: {linewidth: 3, color: '0xe74c3c'}});    // Urethane - Red
        """

    # Molecule colors
    mol_colors = {
        "phenol": "0xe74c3c",
        "m-cresol": "0x9b59b6",
        "glucose": "0xf39c12",
        "oxygen": "0x3498db"
    }

    mol_sphere_js = ""
    animation_js = ""

    if molecule_info:
        color = mol_colors.get(molecule_info['name'], "0x1abc9c")

        if animate:
            animation_js = f"""
            var sphereId = null;
            var startZ = 25;
            var endZ = -25;
            var duration = 4000;
            var startTime = Date.now();
            var color = {color};

            function animatePermeation() {{
                var elapsed = Date.now() - startTime;
                var progress = elapsed / duration;

                if (progress >= 1) {{
                    if (sphereId !== null) viewer.removeShape(sphereId);
                    sphereId = viewer.addSphere({{
                        center: {{x: 0, y: 0, z: 0}},
                        radius: 2.5,
                        color: color,
                        opacity: 0.95
                    }});
                    viewer.render();
                    return;
                }}

                var z;
                if (progress < 0.3) {{
                    z = startZ - startZ * (progress / 0.3);
                }} else if (progress < 0.7) {{
                    var membraneProgress = (progress - 0.3) / 0.4;
                    z = 0 - (endZ * 0.5) * membraneProgress;
                }} else {{
                    var exitProgress = (progress - 0.7) / 0.3;
                    z = endZ * 0.5 - (endZ * 0.5) * exitProgress;
                }}

                if (sphereId !== null) viewer.removeShape(sphereId);
                sphereId = viewer.addSphere({{
                    center: {{x: 0, y: 0, z: z}},
                    radius: 2.5,
                    color: color,
                    opacity: 0.95
                }});
                viewer.render();
                requestAnimationFrame(animatePermeation);
            }}

            animatePermeation();
            """
        else:
            mol_sphere_js = f"""
            viewer.addSphere({{
                center: {{x: 0, y: 0, z: 0}},
                radius: 2.5,
                color: {color},
                opacity: 0.95
            }});
            """

    # Box dimensions for bounding box (must match generate_polymer_chains)
    box_x, box_y, box_z = 40, 40, 15

    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");

        {color_scheme}

        // Draw membrane bounding box (wireframe)
        var boxX = {box_x};
        var boxY = {box_y};
        var boxZ = {box_z};
        var hx = boxX/2, hy = boxY/2, hz = boxZ/2;
        var boxColor = 0x555555;
        var boxWidth = 1.5;

        // Bottom face edges
        viewer.addLine({{start: {{x: -hx, y: -hy, z: -hz}}, end: {{x: hx, y: -hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: -hz}}, end: {{x: hx, y: hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: -hz}}, end: {{x: -hx, y: hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: -hz}}, end: {{x: -hx, y: -hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});

        // Top face edges
        viewer.addLine({{start: {{x: -hx, y: -hy, z: hz}}, end: {{x: hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: hz}}, end: {{x: hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: hz}}, end: {{x: -hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: hz}}, end: {{x: -hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});

        // Vertical edges connecting top and bottom
        viewer.addLine({{start: {{x: -hx, y: -hy, z: -hz}}, end: {{x: -hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: -hz}}, end: {{x: hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: -hz}}, end: {{x: hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: -hz}}, end: {{x: -hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});

        {mol_sphere_js}

        viewer.zoomTo();
        viewer.zoom(0.5);  // Zoom out to show full membrane box
        viewer.rotate(20, {{x: 1, y: 0, z: 0}});
        viewer.rotate(-15, {{x: 0, y: 1, z: 0}});
        viewer.render();

        {animation_js}
    </script>
    """
    components.html(html, height=520)


# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Membrane Composition")

    # Thickness
    thickness = st.number_input("Thickness (um)", value=200, min_value=10, max_value=500, step=10)

    st.markdown("**Sparsa Polymers (%)**")
    c1, c2 = st.columns(2)
    with c1:
        sparsa1_pct = st.number_input("Sparsa 1", value=30, min_value=0, max_value=100, key="sparsa1")
    with c2:
        sparsa2_pct = st.number_input("Sparsa 2", value=0, min_value=0, max_value=100, key="sparsa2")

    st.markdown("**Carbosil Polymers (%)**")
    c3, c4 = st.columns(2)
    with c3:
        carbosil1_pct = st.number_input("Carbosil 1", value=70, min_value=0, max_value=100, key="carbosil1")
    with c4:
        carbosil2_pct = st.number_input("Carbosil 2", value=0, min_value=0, max_value=100, key="carbosil2")

    # Normalize
    total = sparsa1_pct + sparsa2_pct + carbosil1_pct + carbosil2_pct
    if total > 0:
        sparsa1_frac = sparsa1_pct / total
        sparsa2_frac = sparsa2_pct / total
        carbosil1_frac = carbosil1_pct / total
        carbosil2_frac = carbosil2_pct / total
    else:
        sparsa1_frac = 0.25
        sparsa2_frac = 0.25
        carbosil1_frac = 0.25
        carbosil2_frac = 0.25

    # For backwards compatibility, combine into CarboSil and Sparsa
    carbosil_frac = carbosil1_frac + carbosil2_frac
    sparsa_frac = sparsa1_frac + sparsa2_frac

    st.caption(f"Total Sparsa: {sparsa_frac*100:.0f}% | Total Carbosil: {carbosil_frac*100:.0f}%")

    # Build button
    if st.button("Build Membrane", type="primary", use_container_width=True):
        with st.spinner("Building membrane..."):
            try:
                config = TPUMembraneConfig(
                    polymers={
                        "Sparsa1": sparsa1_frac,
                        "Sparsa2": sparsa2_frac,
                        "Carbosil1": carbosil1_frac,
                        "Carbosil2": carbosil2_frac
                    },
                    thickness=float(thickness)
                )

                builder = TPUMembraneBuilder(seed=12345)
                st.session_state.membrane = builder.build(config)
                st.session_state.perm_result = None
                st.success("Membrane built!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Download button
    if st.session_state.membrane:
        report = []
        report.append("TPU Membrane Report")
        report.append("=" * 40)
        report.append(f"Sparsa 1: {sparsa1_frac*100:.1f}%")
        report.append(f"Sparsa 2: {sparsa2_frac*100:.1f}%")
        report.append(f"Carbosil 1: {carbosil1_frac*100:.1f}%")
        report.append(f"Carbosil 2: {carbosil2_frac*100:.1f}%")
        report.append(f"Thickness: {thickness} um")
        props = st.session_state.membrane.properties
        report.append(f"Density: {props.density:.3f} g/cm3")
        report.append(f"Water uptake: {props.water_uptake:.1f}%")

        st.download_button(
            "Download Report",
            "\n".join(report),
            file_name="membrane_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    # Permeability section
    st.subheader("Permeability Calculator")

    # Molecule presets - only the 4 specified
    mol_presets = {
        "Phenol": "phenol",
        "m-Cresol": "m-cresol",
        "Glucose": "glucose",
        "Oxygen": "oxygen"
    }

    selected_mol = st.selectbox("Molecule", list(mol_presets.keys()))

    if st.button("Calculate Permeability", type="primary", use_container_width=True):
        if st.session_state.membrane is None:
            st.error("Build a membrane first")
        else:
            with st.spinner("Calculating..."):
                try:
                    predictor = TPUPermeabilityPredictor(
                        composition=st.session_state.membrane.composition,
                        thickness_um=st.session_state.membrane.thickness
                    )
                    mol_name = mol_presets[selected_mol]
                    result = predictor.calculate_preset(mol_name)

                    st.session_state.perm_result = {
                        'log_p': round(result.log_permeability, 2),
                        'permeability': f"{result.permeability_cm_s:.2e}",
                        'diffusivity': f"{result.diffusivity_cm2_s:.2e}",
                        'solubility': round(result.solubility, 3),
                        'classification': result.classification,
                        'mol_name': mol_name
                    }
                    st.success("Calculated!")
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("3D Viewer")

    if st.session_state.membrane:
        membrane = st.session_state.membrane
        props = membrane.properties

        # Style selector and animate button
        c1, c2 = st.columns([3, 1])
        with c1:
            style = st.radio("Style", ["stick", "sphere", "line"], horizontal=True)
        with c2:
            animate = False
            if st.session_state.perm_result:
                animate = st.button("Animate", use_container_width=True)

        # Molecule info
        mol_info = None
        if st.session_state.perm_result:
            mol_info = {
                'name': st.session_state.perm_result['mol_name']
            }

        # Calculate composition fractions
        comp = membrane.composition
        sparsa_frac = comp.get("Sparsa1", 0) + comp.get("Sparsa2", 0) + comp.get("Sparsa", 0)
        carbosil_frac = comp.get("Carbosil1", 0) + comp.get("Carbosil2", 0) + comp.get("CarboSil", 0)

        # Generate all-atom polymer chain structure
        atoms = generate_polymer_chains(
            membrane,
            sparsa_frac,
            carbosil_frac,
            num_chains=15
        )

        # Render 3D viewer with style support
        render_3dmol_allatom_styled(
            atoms,
            carbosil_frac,
            style,
            mol_info,
            animate
        )

        # Membrane properties
        st.markdown("**Membrane Properties**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Thickness", f"{props.thickness_um} um")
        c2.metric("Density", f"{props.density:.2f} g/cm3")
        c3.metric("Water Uptake", f"{props.water_uptake:.1f}%")
        c4.metric("Free Volume", f"{props.free_volume_fraction:.3f}")
        c5.metric("Soft Seg.", f"{props.soft_segment_fraction*100:.0f}%")

        # Permeability results
        if st.session_state.perm_result:
            st.divider()
            st.markdown("**Permeability Results**")
            res = st.session_state.perm_result
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("log P", res['log_p'])
            c2.metric("P (cm/s)", res['permeability'])
            c3.metric("D (cm2/s)", res['diffusivity'])

            # Classification badge
            class_colors = {"high": "green", "moderate": "orange", "low": "red"}
            c4.markdown(f"**Classification**")
            c4.markdown(f":{class_colors[res['classification']]}[{res['classification'].upper()}]")

    else:
        st.info("Build a membrane to see the 3D structure")
