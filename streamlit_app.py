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


def generate_polymer_chains(membrane, sparsa_frac, carbosil_frac, num_chains=12):
    """
    Generate realistic all-atom polymer chain structures.

    CarboSil structure: -[Si(CH3)2-O]n- (PDMS soft) + -O-CO-O- (polycarbonate) + -NH-CO-O- (urethane hard)
    Sparsa structure: -[CH2-CH2-O]n- (PEG soft) + -NH-CO-O- (urethane hard)
    """
    atoms = []
    atom_id = 1
    res_id = 1

    # Membrane dimensions for positioning chains
    thickness = membrane.properties.thickness_um * 0.01  # Scale to visualization units

    # Bond lengths in Angstroms
    C_C = 1.54
    C_O = 1.43
    C_N = 1.47
    Si_O = 1.64
    Si_C = 1.87
    C_double_O = 1.23

    def add_atom(name, element, x, y, z, res_name):
        nonlocal atom_id, res_id
        atoms.append({
            'id': atom_id,
            'name': name,
            'res_name': res_name,
            'res_id': res_id,
            'x': x, 'y': y, 'z': z,
            'element': element
        })
        atom_id += 1

    def generate_pdms_unit(start_x, start_y, start_z, direction):
        """Generate -[Si(CH3)2-O]- PDMS repeat unit"""
        nonlocal res_id
        x, y, z = start_x, start_y, start_z
        dx, dy, dz = direction

        # Si atom
        add_atom("SI", "SI", x, y, z, "PDM")
        # Methyl carbons on Si
        add_atom("C", "C", x + 1.2, y + 1.0, z, "PDM")
        add_atom("C", "C", x + 1.2, y - 1.0, z, "PDM")
        # O bridging
        x += dx * Si_O
        y += dy * Si_O * 0.5
        z += dz * Si_O * 0.3
        add_atom("O", "O", x, y, z, "PDM")
        res_id += 1
        return x, y, z

    def generate_peg_unit(start_x, start_y, start_z, direction):
        """Generate -[CH2-CH2-O]- PEG repeat unit"""
        nonlocal res_id
        x, y, z = start_x, start_y, start_z
        dx, dy, dz = direction

        # CH2
        add_atom("C", "C", x, y, z, "PEG")
        x += dx * C_C
        y += dy * C_C * 0.3
        # CH2
        add_atom("C", "C", x, y, z, "PEG")
        x += dx * C_O
        y += dy * C_O * 0.2
        z += dz * C_O * 0.4
        # O ether
        add_atom("O", "O", x, y, z, "PEG")
        res_id += 1
        return x, y, z

    def generate_urethane_unit(start_x, start_y, start_z, direction):
        """Generate -NH-CO-O- urethane linkage"""
        nonlocal res_id
        x, y, z = start_x, start_y, start_z
        dx, dy, dz = direction

        # N (from amine)
        add_atom("N", "N", x, y, z, "URE")
        x += dx * C_N
        # C (carbonyl)
        add_atom("C", "C", x, y, z, "URE")
        # O (carbonyl double bond)
        add_atom("O", "O", x, y + 1.2, z, "URE")
        x += dx * C_O
        y += dy * C_O * 0.3
        # O (ester)
        add_atom("O", "O", x, y, z, "URE")
        res_id += 1
        return x, y, z

    def generate_carbonate_unit(start_x, start_y, start_z, direction):
        """Generate -O-CO-O- polycarbonate unit"""
        nonlocal res_id
        x, y, z = start_x, start_y, start_z
        dx, dy, dz = direction

        # O
        add_atom("O", "O", x, y, z, "PCB")
        x += dx * C_O
        # C (carbonyl)
        add_atom("C", "C", x, y, z, "PCB")
        # O (carbonyl)
        add_atom("O", "O", x, y + 1.1, z, "PCB")
        x += dx * C_O
        # O
        add_atom("O", "O", x, y, z, "PCB")
        res_id += 1
        return x, y, z

    # Generate polymer chains distributed across the membrane
    np.random.seed(42)

    for chain_idx in range(num_chains):
        # Random starting position spread across membrane
        start_x = np.random.uniform(-15, -10)
        start_y = np.random.uniform(-20, 20)
        start_z = np.random.uniform(-thickness/2, thickness/2)

        # Chain direction with some randomness
        dir_x = 1.0
        dir_y = np.random.uniform(-0.3, 0.3)
        dir_z = np.random.uniform(-0.2, 0.2)
        direction = (dir_x, dir_y, dir_z)

        x, y, z = start_x, start_y, start_z

        # Determine chain composition based on polymer fractions
        is_carbosil_chain = np.random.random() < carbosil_frac

        # Generate chain with alternating soft and hard segments
        chain_length = np.random.randint(8, 15)

        for segment in range(chain_length):
            # Update direction slightly for realistic chain wiggle
            direction = (
                direction[0],
                direction[1] + np.random.uniform(-0.1, 0.1),
                direction[2] + np.random.uniform(-0.1, 0.1)
            )

            if segment % 3 == 2:
                # Hard segment (urethane linkage)
                x, y, z = generate_urethane_unit(x, y, z, direction)
            else:
                # Soft segment
                if is_carbosil_chain:
                    # CarboSil: PDMS + polycarbonate
                    if segment % 2 == 0:
                        for _ in range(2):  # 2 PDMS units
                            x, y, z = generate_pdms_unit(x, y, z, direction)
                    else:
                        x, y, z = generate_carbonate_unit(x, y, z, direction)
                else:
                    # Sparsa: PEG soft segment
                    for _ in range(3):  # 3 PEG units
                        x, y, z = generate_peg_unit(x, y, z, direction)

            # Stop if chain gets too long
            if x > 25:
                break

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

    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");

        {color_scheme}

        {mol_sphere_js}

        viewer.zoomTo();
        viewer.zoom(0.6);  // Zoom out to show more context
        viewer.rotate(15, {{x: 1, y: 0, z: 0}});
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

    # Style-based rendering with element-based coloring for realistic look
    # URE = Urethane (hard segment) - Red/Orange
    # PDM = PDMS/Silicone (CarboSil soft) - Blue/Cyan
    # PCB = Polycarbonate (CarboSil) - Purple
    # PEG = Polyether (Sparsa soft) - Green

    if style == "stick":
        color_scheme = """
        // Color by element for realistic look
        viewer.setStyle({elem: 'C'}, {stick: {radius: 0.15, color: '0x909090'}});
        viewer.setStyle({elem: 'O'}, {stick: {radius: 0.15, color: '0xe74c3c'}, sphere: {scale: 0.25, color: '0xe74c3c'}});
        viewer.setStyle({elem: 'N'}, {stick: {radius: 0.15, color: '0x3498db'}, sphere: {scale: 0.25, color: '0x3498db'}});
        viewer.setStyle({elem: 'SI'}, {stick: {radius: 0.18, color: '0xf39c12'}, sphere: {scale: 0.3, color: '0xf39c12'}});
        """
    elif style == "sphere":
        color_scheme = """
        // CPK-style coloring
        viewer.setStyle({elem: 'C'}, {sphere: {scale: 0.4, color: '0x909090'}});
        viewer.setStyle({elem: 'O'}, {sphere: {scale: 0.35, color: '0xe74c3c'}});
        viewer.setStyle({elem: 'N'}, {sphere: {scale: 0.35, color: '0x3498db'}});
        viewer.setStyle({elem: 'SI'}, {sphere: {scale: 0.5, color: '0xf39c12'}});
        """
    else:  # line - color by residue type to show polymer segments
        color_scheme = """
        // Color by segment type
        viewer.setStyle({resn: 'URE'}, {line: {linewidth: 3, color: '0xe74c3c'}});  // Urethane - Red
        viewer.setStyle({resn: 'PDM'}, {line: {linewidth: 2, color: '0x3498db'}});  // PDMS - Blue
        viewer.setStyle({resn: 'PCB'}, {line: {linewidth: 2, color: '0x9b59b6'}});  // Polycarbonate - Purple
        viewer.setStyle({resn: 'PEG'}, {line: {linewidth: 2, color: '0x2ecc71'}});  // PEG - Green
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

    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");

        {color_scheme}

        {mol_sphere_js}

        viewer.zoomTo();
        viewer.zoom(0.6);  // Zoom out to show more context
        viewer.rotate(15, {{x: 1, y: 0, z: 0}});
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
