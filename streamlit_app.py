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
    page_icon="🧪",
    layout="wide"
)

st.title("TPU Membrane Builder")

# Initialize session state
if 'membrane' not in st.session_state:
    st.session_state.membrane = None
if 'perm_result' not in st.session_state:
    st.session_state.perm_result = None


def render_membrane_view(structure, carbosil_frac, sparsa_frac, molecule_info=None, animate=False):
    """Render membrane structure visualization with molecule"""

    if structure is None:
        return

    # Take middle slice
    mid_z = structure.shape[2] // 2
    slice_2d = structure[:, :, mid_z]

    nx, ny = slice_2d.shape
    scale = 5
    width = nx * scale
    height = ny * scale

    # Molecule colors
    mol_colors = {
        "phenol": "#e74c3c",
        "m-cresol": "#9b59b6",
        "glucose": "#f39c12",
        "oxygen": "#3498db"
    }

    mol_js = ""
    animation_js = ""

    if molecule_info:
        color = mol_colors.get(molecule_info['name'], "#1abc9c")
        y_pos = height // 2

        if animate:
            animation_js = f"""
            var molY = 0;
            var molColor = '{color}';
            var duration = 4000;
            var startTime = Date.now();
            var midY = {height // 2};

            function animateMol() {{
                var elapsed = Date.now() - startTime;
                var progress = elapsed / duration;

                if (progress >= 1) {{
                    // Reset
                    molY = midY;
                    drawMembrane();
                    drawMolecule(molY);
                    return;
                }}

                // Move from top to bottom with slowdown in middle
                if (progress < 0.3) {{
                    molY = progress / 0.3 * midY * 0.8;
                }} else if (progress < 0.7) {{
                    var p = (progress - 0.3) / 0.4;
                    molY = midY * 0.8 + p * midY * 0.4;
                }} else {{
                    var p = (progress - 0.7) / 0.3;
                    molY = midY * 1.2 + p * (midY * 0.8);
                }}

                drawMembrane();
                drawMolecule(molY);
                requestAnimationFrame(animateMol);
            }}

            function drawMolecule(y) {{
                ctx.beginPath();
                ctx.arc({width // 2}, y, 8, 0, 2 * Math.PI);
                ctx.fillStyle = molColor;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }}

            animateMol();
            """
        else:
            mol_js = f"""
            // Draw molecule at center
            ctx.beginPath();
            ctx.arc({width // 2}, {y_pos}, 8, 0, 2 * Math.PI);
            ctx.fillStyle = '{color}';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
            """

    html = f"""
    <canvas id="membrane-canvas" width="{width}" height="{height}" style="border-radius: 6px; background: #1a1a1a;"></canvas>
    <script>
        var canvas = document.getElementById('membrane-canvas');
        var ctx = canvas.getContext('2d');
        var scale = {scale};
        var data = {slice_2d.tolist()};
        var carbosil_frac = {carbosil_frac};

        function drawMembrane() {{
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, {width}, {height});

            for (var i = 0; i < data.length; i++) {{
                for (var j = 0; j < data[i].length; j++) {{
                    var val = data[i][j];
                    if (val > 0.5) {{
                        // Hard segment - CarboSil is blue, Sparsa is red
                        var r = Math.round(52 * carbosil_frac + 231 * (1-carbosil_frac));
                        var g = Math.round(152 * carbosil_frac + 76 * (1-carbosil_frac));
                        var b = Math.round(219 * carbosil_frac + 60 * (1-carbosil_frac));
                        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                    }} else if (val > 0.2) {{
                        ctx.fillStyle = '#5d6d7e';
                    }} else {{
                        ctx.fillStyle = '#2c3e50';
                    }}
                    ctx.fillRect(i * scale, j * scale, scale, scale);
                }}
            }}
        }}

        drawMembrane();
        {mol_js}
        {animation_js}
    </script>
    """
    components.html(html, height=height + 20)


# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Membrane Composition")

    # Thickness
    thickness = st.number_input("Thickness (µm)", value=100, min_value=10, max_value=500, step=10)

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
        report.append(f"Thickness: {thickness} µm")
        props = st.session_state.membrane.properties
        report.append(f"Density: {props.density:.3f} g/cm³")
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
    st.subheader("Membrane Structure")

    if st.session_state.membrane:
        membrane = st.session_state.membrane
        props = membrane.properties

        # Animate button
        c1, c2 = st.columns([3, 1])
        with c2:
            animate = False
            if st.session_state.perm_result:
                animate = st.button("▶ Animate", use_container_width=True)

        # Molecule info
        mol_info = None
        if st.session_state.perm_result:
            mol_info = {
                'name': st.session_state.perm_result['mol_name']
            }

        # Render membrane
        render_membrane_view(
            membrane.get_structure(),
            membrane.composition.get("CarboSil", 0.5),
            membrane.composition.get("Sparsa", 0.5),
            mol_info,
            animate
        )

        # Membrane properties
        st.markdown("**Membrane Properties**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Thickness", f"{props.thickness_um} µm")
        c2.metric("Density", f"{props.density:.2f} g/cm³")
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
            c3.metric("D (cm²/s)", res['diffusivity'])

            # Classification badge
            class_colors = {"high": "green", "moderate": "orange", "low": "red"}
            c4.markdown(f"**Classification**")
            c4.markdown(f":{class_colors[res['classification']]}[{res['classification'].upper()}]")

    else:
        st.info("Build a membrane to see the structure")
