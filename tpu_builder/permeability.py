"""
Permeability prediction for TPU membranes

Uses solution-diffusion model adapted for polymer membranes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from .polymers import PolymerLibrary, calculate_blend_properties


@dataclass
class MoleculeDescriptor:
    """Descriptor for a permeating molecule"""
    name: str
    molecular_weight: float  # Da
    molar_volume: float  # cm³/mol
    solubility_parameter: float  # (J/cm³)^0.5
    n_hbd: int  # H-bond donors
    n_hba: int  # H-bond acceptors
    log_p: float  # Octanol-water partition coefficient
    charge: float = 0

    @classmethod
    def simple(
        cls,
        name: str,
        molecular_weight: float,
        log_p: float = 0,
        n_hbd: int = 0,
        n_hba: int = 0
    ) -> 'MoleculeDescriptor':
        """Create a molecule with estimated properties"""
        # Estimate molar volume from MW (rough correlation)
        molar_volume = molecular_weight * 0.9  # Approximate

        # Estimate solubility parameter from log_p
        # Hydrophobic molecules have lower solubility parameters
        solubility_parameter = 23 - log_p * 2

        return cls(
            name=name,
            molecular_weight=molecular_weight,
            molar_volume=molar_volume,
            solubility_parameter=solubility_parameter,
            n_hbd=n_hbd,
            n_hba=n_hba,
            log_p=log_p
        )


# Molecule presets - only phenol, m-cresol, glucose, oxygen
MOLECULE_PRESETS = {
    'phenol': MoleculeDescriptor(
        name='phenol',
        molecular_weight=94.11,
        molar_volume=89.0,
        solubility_parameter=24.1,
        n_hbd=1,
        n_hba=1,
        log_p=1.46
    ),
    'm-cresol': MoleculeDescriptor(
        name='m-cresol',
        molecular_weight=108.14,
        molar_volume=105.0,
        solubility_parameter=23.3,
        n_hbd=1,
        n_hba=1,
        log_p=1.96
    ),
    'glucose': MoleculeDescriptor(
        name='glucose',
        molecular_weight=180.16,
        molar_volume=115.0,
        solubility_parameter=35.0,
        n_hbd=5,
        n_hba=6,
        log_p=-3.0
    ),
    'oxygen': MoleculeDescriptor(
        name='oxygen',
        molecular_weight=32.0,
        molar_volume=25.6,
        solubility_parameter=8.2,
        n_hbd=0,
        n_hba=0,
        log_p=0.65
    ),
}


@dataclass
class PermeabilityResult:
    """Result of permeability calculation"""
    molecule_name: str
    permeability_cm_s: float  # cm/s
    log_permeability: float  # log10(P)
    diffusivity_cm2_s: float  # cm²/s
    solubility: float  # dimensionless partition coefficient
    flux_mol_cm2_s: float  # mol/(cm²·s) at 1M gradient
    classification: str  # high, moderate, low


class TPUPermeabilityPredictor:
    """
    Predicts molecular permeability through TPU membranes

    Uses solution-diffusion model:
    P = D * K / L

    where:
    - P = permeability coefficient
    - D = diffusivity in membrane
    - K = partition coefficient (solubility)
    - L = membrane thickness
    """

    # Reference values for normalization
    D_REF = 1e-7  # cm²/s, reference diffusivity
    K_REF = 1.0  # Reference partition coefficient

    def __init__(
        self,
        composition: Optional[Dict[str, float]] = None,
        thickness_um: float = 100.0,
        temperature: float = 310.15  # 37°C in Kelvin
    ):
        self.library = PolymerLibrary()
        self.composition = composition or {"CarboSil": 1.0}
        self.thickness_um = thickness_um
        self.thickness_cm = thickness_um * 1e-4  # Convert to cm
        self.temperature = temperature

        # Calculate blend properties
        self.blend_props = calculate_blend_properties(self.composition, self.library)

    def calculate(
        self,
        molecule: MoleculeDescriptor
    ) -> PermeabilityResult:
        """
        Calculate permeability for a molecule through the membrane

        Args:
            molecule: MoleculeDescriptor for the permeating species

        Returns:
            PermeabilityResult with all calculated values
        """
        # Calculate diffusivity
        D = self._calculate_diffusivity(molecule)

        # Calculate solubility/partition coefficient
        K = self._calculate_solubility(molecule)

        # Permeability from solution-diffusion model
        P = D * K / self.thickness_cm

        # Calculate flux at unit concentration gradient
        flux = P  # mol/(cm²·s) for 1 mol/cm³ gradient

        # Log permeability
        log_P = np.log10(max(P, 1e-20))

        # Classification
        if log_P > -6:
            classification = "high"
        elif log_P > -8:
            classification = "moderate"
        else:
            classification = "low"

        return PermeabilityResult(
            molecule_name=molecule.name,
            permeability_cm_s=P,
            log_permeability=log_P,
            diffusivity_cm2_s=D,
            solubility=K,
            flux_mol_cm2_s=flux,
            classification=classification
        )

    def _calculate_diffusivity(self, molecule: MoleculeDescriptor) -> float:
        """
        Calculate diffusivity using free volume theory

        D = D0 * exp(-Bd * V / Vf)

        where:
        - D0 = reference diffusivity
        - Bd = empirical constant
        - V = molecular volume
        - Vf = free volume of polymer
        """
        # Base diffusivity (Stokes-Einstein scaling)
        MW = molecule.molecular_weight
        D0 = self.D_REF * (100 / MW) ** 0.5

        # Free volume effect
        Vf = self.blend_props['free_volume_fraction']
        Vm = molecule.molar_volume / 100  # Normalize
        Bd = 0.5

        D_fv = D0 * np.exp(-Bd * Vm / max(Vf, 0.01))

        # Crystallinity reduction (tortuous path)
        cryst = self.blend_props['crystallinity']
        tau = 1 + cryst * 2  # Tortuosity factor
        D_cryst = D_fv / tau

        # Temperature dependence (Arrhenius)
        Ea = 20000  # J/mol, activation energy
        R = 8.314
        T_ref = 298.15
        D_temp = D_cryst * np.exp(-Ea / R * (1 / self.temperature - 1 / T_ref))

        return D_temp

    def _calculate_solubility(self, molecule: MoleculeDescriptor) -> float:
        """
        Calculate partition coefficient using solubility parameter theory

        K ~ exp(-V * (δ1 - δ2)² / RT)

        where:
        - V = molar volume
        - δ1, δ2 = solubility parameters of molecule and polymer
        """
        # Effective polymer solubility parameter
        delta_poly = 18.0  # Approximate for TPU blend
        if self.blend_props['hydrophilicity'] > 0.5:
            delta_poly = 22.0  # More hydrophilic

        delta_mol = molecule.solubility_parameter

        # Flory-Huggins interaction parameter
        R = 8.314
        V = molecule.molar_volume * 1e-6  # Convert to m³/mol
        chi = V * (delta_mol - delta_poly) ** 2 / (R * self.temperature)

        # Partition coefficient
        K = np.exp(-chi)

        # Hydrophilicity boost for polar molecules
        hydro = self.blend_props['hydrophilicity']
        if molecule.n_hbd > 0 or molecule.n_hba > 0:
            polar_factor = 1 + hydro * (molecule.n_hbd + molecule.n_hba) * 0.1
            K *= polar_factor

        # Water uptake effect (swollen membrane is more permeable to hydrophilics)
        water = self.blend_props['water_uptake']
        if molecule.log_p < 0:  # Hydrophilic molecule
            K *= (1 + water * 0.05)

        return min(K, 10)  # Cap at reasonable value

    def get_preset_molecule(self, name: str) -> MoleculeDescriptor:
        """Get a preset molecule by name"""
        if name not in MOLECULE_PRESETS:
            raise ValueError(f"Unknown molecule: {name}. Available: {list(MOLECULE_PRESETS.keys())}")
        return MOLECULE_PRESETS[name]

    def calculate_preset(self, molecule_name: str) -> PermeabilityResult:
        """Calculate permeability for a preset molecule"""
        molecule = self.get_preset_molecule(molecule_name)
        return self.calculate(molecule)

    def compare_molecules(self, molecules: list) -> Dict[str, PermeabilityResult]:
        """Calculate permeability for multiple molecules"""
        results = {}
        for mol_name in molecules:
            if isinstance(mol_name, str):
                mol = self.get_preset_molecule(mol_name)
            else:
                mol = mol_name
            results[mol.name] = self.calculate(mol)
        return results
