# TPU Membrane Builder

Build thermoplastic polyurethane (TPU) membranes from CarboSil and Sparsa polymers and predict molecular permeability.

**Live Demo**: https://tpu-membrane-builder.streamlit.app/

## Polymers

### CarboSil
Segmented silicone-polycarbonate polyurethane:
- **Soft segments**: PDMS (polydimethylsiloxane) diol
- **Hard segments**: Aliphatic polycarbonate diol
- **Linkages**: Urethane bonds (HDI or H12MDI-type)
- **Properties**: High mechanical stiffness, lower permeability, reduced swelling

### Sparsa
Amphiphilic polyurethane with flexible soft segment:
- **Soft segment**: Polyether diol (PEG/PPG-like) or very short PDMS diol
- **Hard segment**: Urethane domains (lower content than CarboSil)
- **Properties**: Higher permeability, hydrated but not dissolving, amphiphilic

## Installation

```bash
pip install numpy matplotlib streamlit
```

## Quick Start

```bash
python run.py
```

## Web App

```bash
streamlit run streamlit_app.py
```

## Configuration

Edit `args.txt`:

```
thickness = 100.0  # micrometers

CarboSil = 0.7
Sparsa = 0.3
```

## Python API

### Building Membranes

```python
from tpu_builder import TPUMembraneBuilder, TPUMembraneConfig

# Pure CarboSil
config = TPUMembraneConfig.create_carbosil(thickness=100)

# Pure Sparsa
config = TPUMembraneConfig.create_sparsa(thickness=100)

# Blend
config = TPUMembraneConfig.create_blend(
    carbosil_fraction=0.7,
    sparsa_fraction=0.3,
    thickness=100
)

builder = TPUMembraneBuilder()
membrane = builder.build(config)
membrane.write_report("membrane_report.txt")
```

### Predicting Permeability

```python
from tpu_builder import TPUPermeabilityPredictor

predictor = TPUPermeabilityPredictor(
    composition={"CarboSil": 0.7, "Sparsa": 0.3},
    thickness_um=100
)

# Available molecules: phenol, m-cresol, glucose, oxygen
result = predictor.calculate_preset("phenol")
print(f"log P = {result.log_permeability:.2f}")
print(f"P = {result.permeability_cm_s:.2e} cm/s")
```

## Available Molecules

| Molecule | MW (Da) | log P (octanol/water) |
|----------|---------|----------------------|
| Phenol | 94.11 | 1.46 |
| m-Cresol | 108.14 | 1.96 |
| Glucose | 180.16 | -3.0 |
| Oxygen | 32.0 | 0.65 |

## Membrane Properties

| Property | CarboSil | Sparsa |
|----------|----------|--------|
| Density | 1.05 g/cm³ | 1.08 g/cm³ |
| Water uptake | 0.5% | 8% |
| Free volume | 0.03 | 0.08 |
| Crystallinity | 0.15 | 0.05 |
| Hydrophilicity | 0.2 | 0.6 |
| Soft segment | 65% | 75% |

## Permeability Model

Uses solution-diffusion model:

```
P = D × K / L
```

Where:
- P = permeability coefficient (cm/s)
- D = diffusivity (cm²/s)
- K = partition coefficient
- L = membrane thickness (cm)

## Classification

- **High**: log P > -6
- **Moderate**: -6 > log P > -8
- **Low**: log P < -8

## License

MIT License
