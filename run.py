#!/usr/bin/env python
"""
TPU Membrane Builder - Command Line Interface

Build TPU membranes from CarboSil and Sparsa polymers and predict permeability.
"""

import os
from tpu_builder import (
    TPUMembraneBuilder,
    TPUMembraneConfig,
    TPUPermeabilityPredictor,
    MoleculeDescriptor
)


def parse_args(filepath: str) -> dict:
    """Parse configuration from args.txt"""
    config = {
        'thickness': 100.0,
        'width': 10.0,
        'height': 10.0,
        'polymers': {}
    }

    if not os.path.exists(filepath):
        print(f"No {filepath} found, using defaults")
        config['polymers'] = {'CarboSil': 1.0}
        return config

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove inline comments
                if '#' in value:
                    value = value.split('#')[0].strip()

                if key == 'thickness':
                    config['thickness'] = float(value)
                elif key == 'width':
                    config['width'] = float(value)
                elif key == 'height':
                    config['height'] = float(value)
                elif key in ['CarboSil', 'Sparsa']:
                    config['polymers'][key] = float(value)

    # Normalize polymer fractions
    if config['polymers']:
        total = sum(config['polymers'].values())
        if total > 0:
            for k in config['polymers']:
                config['polymers'][k] /= total
    else:
        config['polymers'] = {'CarboSil': 1.0}

    return config


def main():
    print("=" * 50)
    print("TPU Membrane Builder")
    print("=" * 50)

    # Parse configuration
    args = parse_args('args.txt')

    print(f"\nConfiguration:")
    print(f"  Thickness: {args['thickness']} µm")
    print(f"  Dimensions: {args['width']} x {args['height']} mm")
    print(f"  Polymers:")
    for polymer, fraction in args['polymers'].items():
        print(f"    {polymer}: {fraction*100:.1f}%")

    # Build membrane
    print("\nBuilding membrane...")
    config = TPUMembraneConfig(
        polymers=args['polymers'],
        thickness=args['thickness'],
        width=args['width'],
        height=args['height']
    )

    builder = TPUMembraneBuilder(seed=12345)
    membrane = builder.build(config)

    # Create output directory
    os.makedirs('Outputs', exist_ok=True)

    # Write report
    report_path = 'Outputs/membrane_report.txt'
    membrane.write_report(report_path)
    print(f"  Report saved: {report_path}")

    # Print properties
    props = membrane.properties
    print(f"\nMembrane Properties:")
    print(f"  Density: {props.density:.3f} g/cm³")
    print(f"  Water uptake: {props.water_uptake:.1f}%")
    print(f"  Free volume: {props.free_volume_fraction:.3f}")
    print(f"  Crystallinity: {props.crystallinity:.3f}")
    print(f"  Hydrophilicity: {props.hydrophilicity:.2f}")
    print(f"  Soft segment: {props.soft_segment_fraction*100:.1f}%")
    print(f"  Permeability factor: {props.permeability_factor:.3f}")

    # Calculate permeability for common molecules
    print("\n" + "=" * 50)
    print("Permeability Predictions")
    print("=" * 50)

    predictor = TPUPermeabilityPredictor(
        composition=args['polymers'],
        thickness_um=args['thickness']
    )

    molecules = ['phenol', 'm-cresol', 'glucose', 'oxygen']

    print(f"\n{'Molecule':<12} {'log P':<10} {'P (cm/s)':<12} {'Class':<10}")
    print("-" * 44)

    for mol_name in molecules:
        result = predictor.calculate_preset(mol_name)
        print(f"{result.molecule_name:<12} {result.log_permeability:<10.2f} {result.permeability_cm_s:<12.2e} {result.classification:<10}")

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == '__main__':
    main()
