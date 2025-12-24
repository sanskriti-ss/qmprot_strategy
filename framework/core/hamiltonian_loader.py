"""
Hamiltonian Loader Module

Loads Hamiltonians from:
1. Text files (hamiltonian_xxx.txt) with format: Coefficient\tOperators
2. JSON metadata files (qmprot.json) for molecule properties
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Molecule:
    """Data class for molecule information"""
    abbreviation: str
    name: str
    n_qubits: int
    n_coefficients: int
    reference_energy: float
    hamiltonian_file: str
    n_electrons: Optional[int] = None
    n_orbitals: Optional[int] = None
    charge: int = 0
    spin: int = 0
    basis: str = "sto-3g"
    coordinates: Optional[List[Dict]] = None
    molecular_formula: Optional[str] = None
    
    
@dataclass
class QubitHamiltonian:
    """Data class for qubit Hamiltonian"""
    molecule: Molecule
    coefficients: np.ndarray
    pauli_strings: List[str]
    n_qubits: int
    n_terms: int
    
    def to_pennylane(self):
        """Convert to PennyLane Hamiltonian format"""
        import pennylane as qml
        
        coeffs = []
        ops = []
        
        for coeff, pauli_string in zip(self.coefficients, self.pauli_strings):
            if np.abs(coeff) < 1e-12:
                continue
            
            coeffs.append(coeff)
            
            # Parse Pauli string to PennyLane operators
            pauli_ops = []
            for i, p in enumerate(pauli_string):
                if p == 'X':
                    pauli_ops.append(qml.PauliX(i))
                elif p == 'Y':
                    pauli_ops.append(qml.PauliY(i))
                elif p == 'Z':
                    pauli_ops.append(qml.PauliZ(i))
                # 'I' is identity, skip
            
            if pauli_ops:
                if len(pauli_ops) == 1:
                    ops.append(pauli_ops[0])
                else:
                    ops.append(pauli_ops[0])
                    for op in pauli_ops[1:]:
                        ops[-1] = ops[-1] @ op
            else:
                # All identity - use Identity on qubit 0
                ops.append(qml.Identity(0))
        
        return qml.Hamiltonian(coeffs, ops)
    
    def to_qiskit(self):
        """Convert to Qiskit SparsePauliOp format"""
        from qiskit.quantum_info import SparsePauliOp
        
        # Qiskit uses reverse qubit ordering
        pauli_labels = [ps[::-1] for ps in self.pauli_strings]
        
        return SparsePauliOp.from_list(
            [(label, coeff) for label, coeff in zip(pauli_labels, self.coefficients)
             if np.abs(coeff) >= 1e-12]
        )
    
    def to_openfermion(self):
        """Convert to OpenFermion QubitOperator format"""
        from openfermion import QubitOperator
        
        hamiltonian = QubitOperator()
        
        for coeff, pauli_string in zip(self.coefficients, self.pauli_strings):
            if np.abs(coeff) < 1e-12:
                continue
            
            term = []
            for i, p in enumerate(pauli_string):
                if p != 'I':
                    term.append((i, p))
            
            hamiltonian += QubitOperator(tuple(term), coeff)
        
        return hamiltonian


class HamiltonianLoader:
    """Load and parse Hamiltonians from files or QMProt datasets"""

    def __init__(self, 
                 hamiltonians_dir: Union[str, Path],
                 molecules_json: Optional[Union[str, Path]] = None):
        """
        Initialize the Hamiltonian loader.
        Args:
            hamiltonians_dir: Directory containing Hamiltonian .txt files or datasets
            molecules_json: Path to QMProt.json metadata file
        """
        self.hamiltonians_dir = Path(hamiltonians_dir)
        self.molecules_json = Path(molecules_json) if molecules_json else None
        self.molecules: Dict[str, Molecule] = {}

        if self.molecules_json and self.molecules_json.exists():
            self._load_molecules_metadata()
    
    def _load_molecules_metadata(self):
        """Load molecule metadata from JSON file"""
        try:
            with open(self.molecules_json, 'r') as f:
                data = json.load(f)
            
            # Load test molecules (H2, H2O, etc.)
            for mol in data.get("test_molecules", []):
                self._add_molecule_from_dict(mol)
            
            # Load amino acids
            for mol in data.get("amino_acids", []):
                self._add_molecule_from_dict(mol)
            
            # Load other molecules
            for mol in data.get("other_molecules", []):
                self._add_molecule_from_dict(mol)
                
            logger.info(f"Loaded metadata for {len(self.molecules)} molecules")
            
        except Exception as e:
            logger.error(f"Error loading molecules JSON: {e}")
    
    def _add_molecule_from_dict(self, mol_dict: Dict):
        """Add a molecule from dictionary data"""
        abbrev = mol_dict.get("abbreviation", "")
        if not abbrev:
            return
            
        self.molecules[abbrev] = Molecule(
            abbreviation=abbrev,
            name=mol_dict.get("name", ""),
            n_qubits=mol_dict.get("n_qubits", 0),
            n_coefficients=mol_dict.get("n_coefficients", 0),
            reference_energy=mol_dict.get("energy", 0.0),
            hamiltonian_file=mol_dict.get("hamiltonian", f"hamiltonian_{abbrev}.txt"),
            n_electrons=mol_dict.get("n_electrons"),
            n_orbitals=mol_dict.get("n_orbitals"),
            charge=mol_dict.get("charge", 0),
            spin=mol_dict.get("spin", 0),
            basis=mol_dict.get("basis", "sto-3g"),
            coordinates=mol_dict.get("coordinates"),
            molecular_formula=mol_dict.get("mf"),
        )
    
    def load_hamiltonian(self, 
                         molecule_abbrev: Optional[str] = None,
                         hamiltonian_file: Optional[Union[str, Path]] = None) -> QubitHamiltonian:
        """
        Load a Hamiltonian from file or QMProt dataset if in datasets/ mode.
        """
        import pennylane as qml
        import os
        # Detect if using datasets/ (QMProt mode)
        if self.hamiltonians_dir.name == "datasets":
            # Expect molecule_abbrev to match a subfolder (e.g., "ala", "gly")
            if molecule_abbrev is None:
                raise ValueError("Must provide molecule_abbrev for QMProt dataset mode.")
            dataset_path = self.hamiltonians_dir / molecule_abbrev / f"{molecule_abbrev}.h5"
            if not dataset_path.exists():
                raise FileNotFoundError(f"QMProt dataset file not found: {dataset_path}")
            ds = qml.data.load("other", name=molecule_abbrev)
            # Find hamiltonian chunks as in hamiltonian_download.ipynb
            hamiltonian_chunks = []
            if isinstance(ds, list):
                dataset = ds[0] if len(ds) > 0 else None
                if dataset is not None:
                    if hasattr(dataset, 'list_attributes'):
                        for key in dataset.list_attributes():
                            if "hamiltonian" in key:
                                hamiltonian_chunks.append(getattr(dataset, key))
                    elif hasattr(dataset, '__dict__'):
                        for key in dir(dataset):
                            if "hamiltonian" in key and not key.startswith('_'):
                                hamiltonian_chunks.append(getattr(dataset, key))
            else:
                for key in ds.list_attributes():
                    if "hamiltonian" in key:
                        hamiltonian_chunks.append(getattr(ds, key))
            if not hamiltonian_chunks:
                raise ValueError("No hamiltonian chunks found in QMProt dataset.")
            full_hamiltonian = "".join(hamiltonian_chunks)
            lines = full_hamiltonian.split("\n")
            valid_lines = [line.strip() for line in lines if line.strip() and "Coefficient" not in line and "Operators" not in line]
            coefficients = []
            pauli_strings = []
            for line in valid_lines:
                parts = line.split()
                try:
                    coeff = float(parts[0])
                    pauli_str = parts[1].strip()
                    coefficients.append(coeff)
                    pauli_strings.append(pauli_str)
                except Exception:
                    continue
            n_qubits = len(pauli_strings[0]) if pauli_strings else 0
            molecule = Molecule(
                abbreviation=molecule_abbrev,
                name=molecule_abbrev,
                n_qubits=n_qubits,
                n_coefficients=len(coefficients),
                reference_energy=0.0,
                hamiltonian_file=str(dataset_path),
            )
            return QubitHamiltonian(
                molecule=molecule,
                coefficients=np.array(coefficients),
                pauli_strings=pauli_strings,
                n_qubits=n_qubits,
                n_terms=len(coefficients),
            )
        
        # Default: text file mode
        if molecule_abbrev:
            # Look up molecule metadata
            if molecule_abbrev not in self.molecules:
                raise ValueError(f"Molecule '{molecule_abbrev}' not found in metadata. "
                                f"Available: {list(self.molecules.keys())}")
            
            molecule = self.molecules[molecule_abbrev]
            file_path = self.hamiltonians_dir / molecule.hamiltonian_file
            
        elif hamiltonian_file:
            # Direct file path provided
            file_path = Path(hamiltonian_file)
            if not file_path.is_absolute():
                file_path = self.hamiltonians_dir / file_path
            
            # Create minimal molecule info
            molecule = Molecule(
                abbreviation=file_path.stem.replace("hamiltonian_", ""),
                name="Unknown",
                n_qubits=0,  # Will be determined from file
                n_coefficients=0,  # Will be determined from file
                reference_energy=0.0,
                hamiltonian_file=str(file_path),
            )
        else:
            raise ValueError("Must provide either molecule_abbrev or hamiltonian_file")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Hamiltonian file not found: {file_path}")
        
        logger.info(f"Loading Hamiltonian from {file_path}")
        
        # Parse the file
        coefficients, pauli_strings = self._parse_hamiltonian_file(file_path)
        
        if not pauli_strings:
            raise ValueError(f"No valid Hamiltonian terms found in {file_path}")
        
        # Determine number of qubits from Pauli strings
        n_qubits = len(pauli_strings[0]) if pauli_strings else 0
        
        # Update molecule info with actual values
        molecule.n_qubits = n_qubits
        molecule.n_coefficients = len(coefficients)
        
        return QubitHamiltonian(
            molecule=molecule,
            coefficients=np.array(coefficients),
            pauli_strings=pauli_strings,
            n_qubits=n_qubits,
            n_terms=len(coefficients),
        )
    
    def _parse_hamiltonian_file(self, file_path: Path) -> Tuple[List[float], List[str]]:
        """
        Parse a Hamiltonian text file.
        
        Expected format:
        Coefficient\tOperators
        0.123456\tIIII
        -0.234567\tZIIZ
        ...
        
        Returns:
            Tuple of (coefficients, pauli_strings)
        """
        coefficients = []
        pauli_strings = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present
        start_idx = 0
        if lines and ('Coefficient' in lines[0] or 'coefficient' in lines[0].lower()):
            start_idx = 1
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    coeff = float(parts[0])
                    pauli_str = parts[1].strip()
                    
                    coefficients.append(coeff)
                    pauli_strings.append(pauli_str)
                except ValueError:
                    continue
        
        logger.info(f"Loaded {len(coefficients)} terms from {file_path.name}")
        return coefficients, pauli_strings
    
    def load_multiple(self, 
                      molecule_abbrevs: Optional[List[str]] = None,
                      hamiltonian_files: Optional[List[Union[str, Path]]] = None) -> List[QubitHamiltonian]:
        """
        Load multiple Hamiltonians.
        
        Args:
            molecule_abbrevs: List of molecule abbreviations
            hamiltonian_files: List of Hamiltonian file paths
            
        Returns:
            List of QubitHamiltonian objects
        """
        hamiltonians = []
        
        if molecule_abbrevs:
            for abbrev in molecule_abbrevs:
                try:
                    h = self.load_hamiltonian(molecule_abbrev=abbrev)
                    hamiltonians.append(h)
                except FileNotFoundError as e:
                    logger.warning(f"Could not load {abbrev}: {e}")
        
        if hamiltonian_files:
            for file_path in hamiltonian_files:
                try:
                    h = self.load_hamiltonian(hamiltonian_file=file_path)
                    hamiltonians.append(h)
                except FileNotFoundError as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        return hamiltonians
    
    def list_available_molecules(self) -> List[str]:
        """List all molecules with metadata"""
        return list(self.molecules.keys())
    
    def list_available_hamiltonians(self) -> List[str]:
        """List all Hamiltonian files in the directory"""
        if not self.hamiltonians_dir.exists():
            return []
        return [f.stem.replace("hamiltonian_", "") 
                for f in self.hamiltonians_dir.glob("hamiltonian_*.txt")]
    
    def get_molecule_info(self, abbreviation: str) -> Optional[Molecule]:
        """Get molecule information by abbreviation"""
        return self.molecules.get(abbreviation)
