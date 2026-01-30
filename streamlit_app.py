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


def generate_pdb_from_structure(membrane, carbosil_frac):
    """Generate PDB-like atom data for all-atom visualization"""
    structure = membrane.get_structure()
    atoms = []

    nx, ny, nz = structure.shape
    scale = 1.5  # Angstrom-like scaling

    # Generate atoms representing the polymer structure
    atom_id = 1
    res_id = 1

    # Sample with finer resolution for all-atom look
    step = 1

    for i in range(0, nx, step):
        for j in range(0, ny, step):
            for k in range(0, nz, step):
                val = structure[i, j, k]
                if val > 0.05:  # Include more atoms
                    x = (i - nx/2) * scale
                    y = (j - ny/2) * scale
                    z = (k - nz/2) * scale

                    # Assign atom types based on segment type and position
                    if val > 0.6:
                        # Hard segment - urethane linkages
                        if (i + j + k) % 3 == 0:
                            atom_type = "N"  # Nitrogen in urethane
                            element = "N"
                        elif (i + j + k) % 3 == 1:
                            atom_type = "C"  # Carbon
                            element = "C"
                        else:
                            atom_type = "O"  # Oxygen in urethane
                            element = "O"
                        res_name = "URE"  # Urethane
                    elif val > 0.3:
                        # Soft segment
                        if carbosil_frac > 0.5:
                            # PDMS (silicone) - Si, O, C
                            if (i + j) % 3 == 0:
                                atom_type = "SI"
                                element = "SI"
                            elif (i + j) % 3 == 1:
                                atom_type = "O"
                                element = "O"
                            else:
                                atom_type = "C"
                                element = "C"
                            res_name = "PDM"  # PDMS
                        else:
                            # Polyether (PEG-like) - C, O
                            if (i + j) % 2 == 0:
                                atom_type = "C"
                                element = "C"
                            else:
                                atom_type = "O"
                                element = "O"
                            res_name = "PEG"  # Polyether
                    else:
                        # Interface/amorphous region
                        atom_type = "C"
                        element = "C"
                        res_name = "AMO"

                    atoms.append({
                        'id': atom_id,
                        'name': atom_type,
                        'res_name': res_name,
                        'res_id': res_id,
                        'x': x,
                        'y': y,
                        'z': z,
                        'element': element,
                        'val': val
                    })
                    atom_id += 1

                    if atom_id % 50 == 0:
                        res_id += 1

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
    thickness = st.number_input("Thickness (um)", value=100, min_value=10, max_value=500, step=10)

    st.markdown("**Polymers (%)**")

    # Polymer inputs
    c1, c2 = st.columns(2)
    with c1:
        carbosil_pct = st.number_input("CarboSil", value=70, min_value=0, max_value=100, key="carbosil")
    with c2:
        sparsa_pct = st.number_input("Sparsa", value=30, min_value=0, max_value=100, key="sparsa")

    # Normalize
    total = carbosil_pct + sparsa_pct
    if total > 0:
        carbosil_frac = carbosil_pct / total
        sparsa_frac = sparsa_pct / total
    else:
        carbosil_frac = 0.5
        sparsa_frac = 0.5

    st.caption(f"Normalized: CarboSil {carbosil_frac*100:.0f}%, Sparsa {sparsa_frac*100:.0f}%")

    # Build button
    if st.button("Build Membrane", type="primary", use_container_width=True):
        with st.spinner("Building membrane..."):
            try:
                config = TPUMembraneConfig(
                    polymers={
                        "CarboSil": carbosil_frac,
                        "Sparsa": sparsa_frac
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
        report.append(f"CarboSil: {carbosil_frac*100:.1f}%")
        report.append(f"Sparsa: {sparsa_frac*100:.1f}%")
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

        # Generate all-atom structure
        atoms = generate_pdb_from_structure(
            membrane,
            membrane.composition.get("CarboSil", 0.5)
        )

        # Render 3D viewer with style support
        render_3dmol_allatom_styled(
            atoms,
            membrane.composition.get("CarboSil", 0.5),
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
    if style == "stick":
        if carbosil_frac > 0.5:
            color_scheme = """
            viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.12, color: '0x3498db'}, sphere: {scale: 0.2, color: '0x3498db'}});
            viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.1, color: '0x1abc9c'}, sphere: {scale: 0.18, color: '0x1abc9c'}});
            viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.1, color: '0x2ecc71'}, sphere: {scale: 0.18, color: '0x2ecc71'}});
            viewer.setStyle({resn: 'AMO'}, {stick: {radius: 0.08, color: '0x7f8c8d'}, sphere: {scale: 0.15, color: '0x7f8c8d'}});
            """
        else:
            color_scheme = """
            viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.12, color: '0xe74c3c'}, sphere: {scale: 0.2, color: '0xe74c3c'}});
            viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.1, color: '0x9b59b6'}, sphere: {scale: 0.18, color: '0x9b59b6'}});
            viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.1, color: '0xf39c12'}, sphere: {scale: 0.18, color: '0xf39c12'}});
            viewer.setStyle({resn: 'AMO'}, {stick: {radius: 0.08, color: '0x7f8c8d'}, sphere: {scale: 0.15, color: '0x7f8c8d'}});
            """
    elif style == "sphere":
        if carbosil_frac > 0.5:
            color_scheme = """
            viewer.setStyle({resn: 'URE'}, {sphere: {scale: 0.35, color: '0x3498db'}});
            viewer.setStyle({resn: 'PDM'}, {sphere: {scale: 0.3, color: '0x1abc9c'}});
            viewer.setStyle({resn: 'PEG'}, {sphere: {scale: 0.3, color: '0x2ecc71'}});
            viewer.setStyle({resn: 'AMO'}, {sphere: {scale: 0.25, color: '0x7f8c8d'}});
            """
        else:
            color_scheme = """
            viewer.setStyle({resn: 'URE'}, {sphere: {scale: 0.35, color: '0xe74c3c'}});
            viewer.setStyle({resn: 'PDM'}, {sphere: {scale: 0.3, color: '0x9b59b6'}});
            viewer.setStyle({resn: 'PEG'}, {sphere: {scale: 0.3, color: '0xf39c12'}});
            viewer.setStyle({resn: 'AMO'}, {sphere: {scale: 0.25, color: '0x7f8c8d'}});
            """
    else:  # line
        if carbosil_frac > 0.5:
            color_scheme = """
            viewer.setStyle({resn: 'URE'}, {line: {linewidth: 2, color: '0x3498db'}});
            viewer.setStyle({resn: 'PDM'}, {line: {linewidth: 1.5, color: '0x1abc9c'}});
            viewer.setStyle({resn: 'PEG'}, {line: {linewidth: 1.5, color: '0x2ecc71'}});
            viewer.setStyle({resn: 'AMO'}, {line: {linewidth: 1, color: '0x7f8c8d'}});
            """
        else:
            color_scheme = """
            viewer.setStyle({resn: 'URE'}, {line: {linewidth: 2, color: '0xe74c3c'}});
            viewer.setStyle({resn: 'PDM'}, {line: {linewidth: 1.5, color: '0x9b59b6'}});
            viewer.setStyle({resn: 'PEG'}, {line: {linewidth: 1.5, color: '0xf39c12'}});
            viewer.setStyle({resn: 'AMO'}, {line: {linewidth: 1, color: '0x7f8c8d'}});
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
        viewer.rotate(15, {{x: 1, y: 0, z: 0}});
        viewer.render();

        {animation_js}
    </script>
    """
    components.html(html, height=520)
