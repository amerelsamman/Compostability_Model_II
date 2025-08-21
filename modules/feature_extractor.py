"""
Module for extracting molecular features from SMILES for machine learning prediction
"""

from rdkit import Chem
from rdkit.Chem import rdchem

class FeatureExtractor:
    """Extract features from molecular structures for permeability prediction."""
    
    @staticmethod
    def is_in_ring(mol, atom_idx):
        """Check if an atom is part of any ring."""
        return mol.GetAtomWithIdx(atom_idx).IsInRing()

    @staticmethod
    def get_ring_info(mol):
        """Extract ring features from molecule.
        Identifies and counts different types of rings including:
        - Phenyls (aromatic 6-member carbon rings)
        - Cyclopentanes, cyclohexanes (aliphatic rings)
        - Thiophenes, and other heteroaromatic rings
        """
        # Get all rings
        rings = mol.GetRingInfo()
        
        # Initialize counters
        ring_features = {
            'phenyls': 0,                    # 6-membered aromatic carbon rings
            'cyclohexanes': 0,               # 6-membered saturated carbon rings
            'cyclopentanes': 0,              # 5-membered saturated carbon rings
            'cyclopentenes': 0,              # 5-membered rings with one double bond (non-aromatic)
            'thiophenes': 0,                 # 5-membered aromatic rings with sulfur
            'aromatic_rings_with_n': 0,      # Aromatic rings containing nitrogen
            'aromatic_rings_with_o': 0,      # Aromatic rings containing oxygen
            'aromatic_rings_with_n_o': 0,    # Aromatic rings containing both N and O
            'aromatic_rings_with_s': 0,      # Aromatic rings containing sulfur (excluding thiophenes)
            'aliphatic_rings_with_n': 0,     # Saturated rings containing nitrogen
            'aliphatic_rings_with_o': 0,     # Saturated rings containing oxygen
            'aliphatic_rings_with_n_o': 0,   # Saturated rings containing both N and O
            'aliphatic_rings_with_s': 0,     # Saturated rings containing sulfur
            'other_rings': 0                 # Any other rings
        }
        
        # Get all rings
        for ring in rings.AtomRings():
            # Get atoms in the ring
            ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
            
            # Count atoms of different types in the ring
            carbon_count = sum(1 for atom in ring_atoms if atom.GetSymbol() == 'C')
            nitrogen_count = sum(1 for atom in ring_atoms if atom.GetSymbol() == 'N')
            oxygen_count = sum(1 for atom in ring_atoms if atom.GetSymbol() == 'O')
            sulfur_count = sum(1 for atom in ring_atoms if atom.GetSymbol() == 'S')
            
            # Check if ring is aromatic
            is_aromatic = all(atom.GetIsAromatic() for atom in ring_atoms)
            
            # Check for cyclopentene (5-membered ring with one double bond, not aromatic)
            if len(ring) == 5 and carbon_count == 5 and not is_aromatic:
                # print(f"\nChecking 5-membered ring: {ring}")
                # Count double bonds in the ring
                double_bond_count = 0
                ring_bonds = []
                # First collect all bonds in the ring
                for i in range(len(ring)):
                    atom1_idx = ring[i]
                    atom2_idx = ring[(i + 1) % len(ring)]
                    bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                    if bond is not None:
                        ring_bonds.append(bond)
                        # print(f"Found bond between atoms {atom1_idx} and {atom2_idx}, type: {bond.GetBondType()}")
                
                # Now count double bonds and aromatic bonds
                for bond in ring_bonds:
                    if bond.GetBondType() == Chem.BondType.DOUBLE or \
                       bond.GetBondType() == Chem.BondType.AROMATIC:
                        double_bond_count += 1
                
                # print(f"Found {double_bond_count} double/aromatic bonds in ring")
                
                # If we have 2 aromatic bonds, it's equivalent to 1 double bond in a cyclopentene
                if double_bond_count == 2:
                    # print("Detected cyclopentene")
                    ring_features['cyclopentenes'] += 1
                    continue
                elif double_bond_count == 0:
                    # print("Detected cyclopentane")
                    ring_features['cyclopentanes'] += 1
                    continue
            
            # Check for phenyl (6-membered aromatic ring with 6 carbons)
            if len(ring) == 6 and carbon_count == 6 and is_aromatic:
                ring_features['phenyls'] += 1
                continue
            
            # Check for cyclohexane (6-membered ring with 6 carbons, not aromatic)
            if len(ring) == 6 and carbon_count == 6 and not is_aromatic:
                ring_features['cyclohexanes'] += 1
                continue
            
            # Check for cyclopentane (5-membered ring with 5 carbons, not aromatic)
            if len(ring) == 5 and carbon_count == 5 and not is_aromatic:
                ring_features['cyclopentanes'] += 1
                continue
            
            # Check for thiophene (5-membered aromatic ring with 4 carbons and 1 sulfur)
            if len(ring) == 5 and carbon_count == 4 and sulfur_count == 1 and is_aromatic:
                ring_features['thiophenes'] += 1
                continue
            
            # Check if this is an imide ring (5-membered ring with N and two C=O)
            is_imide = False
            if len(ring) == 5 and nitrogen_count == 1 and oxygen_count == 2:
                # Check for the C=O groups
                carbonyl_count = 0
                for atom in ring_atoms:
                    if atom.GetSymbol() == 'C':
                        for bond in atom.GetBonds():
                            if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == 'O':
                                carbonyl_count += 1
                if carbonyl_count == 2:
                    is_imide = True
                    continue  # Skip counting as aliphatic_rings_with_n
            
            # Classify other rings based on aromaticity and heteroatoms
            if is_aromatic:
                if nitrogen_count > 0 and oxygen_count > 0:
                    ring_features['aromatic_rings_with_n_o'] += 1
                elif nitrogen_count > 0:
                    ring_features['aromatic_rings_with_n'] += 1
                elif oxygen_count > 0:
                    ring_features['aromatic_rings_with_o'] += 1
                elif sulfur_count > 0:
                    ring_features['aromatic_rings_with_s'] += 1
                else:
                    ring_features['other_rings'] += 1
            else:  # Aliphatic rings
                if nitrogen_count > 0 and oxygen_count > 0:
                    ring_features['aliphatic_rings_with_n_o'] += 1
                elif nitrogen_count > 0:
                    ring_features['aliphatic_rings_with_n'] += 1
                elif oxygen_count > 0:
                    ring_features['aliphatic_rings_with_o'] += 1
                elif sulfur_count > 0:
                    ring_features['aliphatic_rings_with_s'] += 1
                else:
                    ring_features['other_rings'] += 1
        
        return ring_features

    @staticmethod
    def get_hybridization_features(smiles):
        """Compute hybridization features for a given SMILES string.
        Calculates fraction of atoms with different hybridization states (SP, SP2, SP3)
        for each element type (C, N, O, etc.)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Define all possible hybridization states for each element
        all_hybridization_features = [
            'SP_C', 'SP_N',  # sp hybridization
            'SP2_C', 'SP2_N', 'SP2_O', 'SP2_S', 'SP2_B',  # sp2 hybridization
            'SP3_C', 'SP3_N', 'SP3_O', 'SP3_S', 'SP3_P', 'SP3_Si', 'SP3_B',  # sp3 hybridization
            'SP3_F', 'SP3_Cl', 'SP3_Br', 'SP3_I',  # halogens
            'SP3D2_S'  # sp3d2 hybridization
        ]
        
        # Initialize all features to zero
        features = {feature: 0 for feature in all_hybridization_features}
        
        # Define elements of interest
        elements = ['C', 'O', 'N', 'Cl', 'F', 'Br', 'S', 'P', 'B', 'I', 'Si']
        
        # First pass: identify atoms affected by [*]
        affected_atoms = set()
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                # Only mark atoms directly bonded to [*] as affected
                for neighbor in atom.GetNeighbors():
                    affected_atoms.add(neighbor.GetIdx())
                # For carbons, check if they're in a chain terminated by [*]
                if neighbor.GetSymbol() == 'C':
                    # If this carbon has only one other neighbor (besides [*]), it's in a chain
                    other_neighbors = [n for n in neighbor.GetNeighbors() if n.GetSymbol() != '*']
                    if len(other_neighbors) == 1:
                        affected_atoms.add(other_neighbors[0].GetIdx())
    
        # Count total atoms for fraction calculation
        total_atoms = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() in elements])
        
        # Iterate through atoms
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            if element not in elements:
                continue
            
            # Get hybridization state
            hybrid = atom.GetHybridization()
            hybrid_str = str(hybrid).split('.')[-1]  # Convert to string like 'SP2'
            
            # Special case for carbonyl carbons - always SP2
            if element == 'C':
                is_carbonyl = False
                for bond in atom.GetBonds():
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        other_atom = bond.GetOtherAtom(atom)
                        if other_atom.GetSymbol() == 'O':
                            is_carbonyl = True
                            break
                if is_carbonyl:
                    hybrid_str = 'SP2'
            
            # Check if atom is affected by [*]
            if atom.GetIdx() in affected_atoms:
                # Only force SP3 for non-aromatic atoms and atoms not in double bonds
                if not atom.GetIsAromatic():
                    has_double_bond = False
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            has_double_bond = True
                            break
                    if not has_double_bond:
                        hybrid_str = 'SP3'  # Count as SP3 due to proximity to [*]
            
            # Special case for nitrogen: check bond types
            if element == 'N' and hybrid_str != 'SP3':  # Only check if not already set to SP3 by [*]
                has_triple_bond = False
                has_double_bond = False
                is_aromatic = atom.GetIsAromatic()
                
                # Check for triple bonds first
                for bond in atom.GetBonds():
                    if bond.GetBondType() == Chem.BondType.TRIPLE:
                        has_triple_bond = True
                        break
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        has_double_bond = True
                
                if has_triple_bond:
                    hybrid_str = 'SP'  # Triple bond (e.g., nitrile)
                elif has_double_bond or is_aromatic:
                    hybrid_str = 'SP2'  # Double bond or aromatic (e.g., imine, pyridine)
                else:
                    hybrid_str = 'SP3'  # Single bonds only
            
            # Special case for oxygen: check number of bonds
            if element == 'O' and hybrid_str != 'SP3':  # Only check if not already set to SP3 by [*]
                num_bonds = len(atom.GetBonds())
                if num_bonds == 1:
                    hybrid_str = 'SP2'  # Single bond (e.g., carbonyl oxygen)
                elif num_bonds == 2:
                    hybrid_str = 'SP3'  # Two bonds (e.g., alcohol, ether)
                else:
                    # print(f"Warning: Oxygen atom with {num_bonds} bonds in molecule {smiles}")
                    pass
            
            # Special case for sulfur: check number of bonds
            if element == 'S' and hybrid_str != 'SP3':  # Only check if not already set to SP3 by [*]
                num_bonds = len(atom.GetBonds())
                if num_bonds == 1:
                    hybrid_str = 'SP2'  # Single bond (e.g., thiol)
                elif num_bonds == 2:
                    hybrid_str = 'SP3'  # Two bonds (e.g., thioether)
                else:
                    # print(f"Warning: Sulfur atom with {num_bonds} bonds in molecule {smiles}")
                    pass
            
            # Create feature name and increment if it's a valid feature
            feature_name = f"{hybrid_str}_{element}"
            if feature_name in features:  # Only increment if it's a valid feature
                features[feature_name] += 1
            else:
                # print(f"Warning: Unexpected hybridization feature found: {feature_name} in molecule {smiles}")
                pass
        
        # Return absolute counts instead of fractions
        for feature in all_hybridization_features:  # Use predefined list to ensure all features are present
            if feature not in features:
                features[feature] = 0
        
        return features

    @staticmethod
    def get_functional_groups(mol):
        """Detect complex functional groups in the molecule.
            Uses a hierarchical approach to identify groups such as:
            - Carboxylic acids, esters, amides
            - Alcohols, ethers, amines
            - And many other functional groups
            """
        if mol is None:
            return None
        
        # Initialize counters for all functional groups
        features = {
            'carboxylic_acid': 0, 'anhydride': 0, 'acyl_halide': 0, 'carbamide': 0,
            'urea': 0, 'carbamate': 0, 'thioamide': 0, 'amide': 0, 'ester': 0,
            'sulfonamide': 0, 'sulfone': 0, 'sulfoxide': 0, 'phosphate': 0,
            'nitro': 0, 'acetal': 0, 'ketal': 0, 'isocyanate': 0, 'thiocyanate': 0,
            'azide': 0, 'azo': 0, 'imide': 0, 'sulfonyl_halide': 0, 'phosphonate': 0,
            'thiourea': 0, 'guanidine': 0, 'silicon_4_coord': 0, 'boron_3_coord': 0,
            'vinyl': 0, 'vinyl_halide': 0, 'allene': 0, 'alcohol': 0, 'ether': 0,
            'aldehyde': 0, 'ketone': 0, 'thiol': 0, 'thioether': 0,
            'primary_amine': 0, 'secondary_amine': 0, 'tertiary_amine': 0, 'quaternary_amine': 0,
            'imine': 0, 'nitrile': 0,
            # Carbon classification (non-ring carbons only)
            'primary_carbon': 0,    # Carbon with 1 non-H neighbor
            'secondary_carbon': 0,  # Carbon with 2 non-H neighbors
            'tertiary_carbon': 0,   # Carbon with 3 non-H neighbors
            'quaternary_carbon': 0  # Carbon with 4 non-H neighbors
        }
        
        # Higher priority numbers are checked first
        patterns = [
            # Nitro groups (highest priority)
            (200, 'nitro', '[#6,*][N+](=[O])[O-]'),  # Charged form
            (199, 'nitro', '[#6,*][N](=O)(=O)'),     # Neutral form
            
            # Complex groups
            (150, 'imide', '[NX3]1([#6,#1,*])[CX3](=[OX1])[#6][CX3](=[OX1])1'),
            (149, 'imide', '[CX3](=[OX1])[NX3][CX3](=[OX1])'),
            (148, 'imide', 'n1c(=O)c2cc3c(=O)n([*])c(=O)c3cc2c1=O'),  # Specific pattern for cyclic imide with [*]
            (147, 'guanidine', '[NX3][CX3](=[NX2])[NX3]'),
            (146, 'thiourea', '[NX3][CX3](=[SX1])[NX3]'),
            (145, 'carbamide', '[NX3][CX3](=[OX1])[NX3]'),
            (144, 'urea', '[NX3H2][CX3](=[OX1])[NX3H2]'),
            (143, 'carbamate', '[NX3][CX3](=[OX1])[OX2][#6,*]'),
            (142, 'thioamide', '[NX3][CX3](=[SX1])[#6,*]'),
            (141, 'amide', '[NX3;H2,H1,H0][CX3](=[OX1])[#6,*]'),
            (140, 'azide', '[NX2]=[NX2]=[NX1]'),
            (139, 'azo', '[NX2]=[NX2]'),
            (138, 'isocyanate', '[NX2]=[CX2]=[OX1]'), 
            
            # Carbonyl-containing groups
            (137, 'anhydride', '[CX3](=[OX1])[OX2][CX3](=[OX1])'),
            (136, 'carboxylic_acid', '[CX3](=[OX1])[OX2H1]'),
            (135, 'acyl_halide', '[CX3](=[OX1])[F,Cl,Br,I]'),
            (134, 'ester', '[CX3](=[OX1])[OX2][#6,*]'),
            (132, 'aldehyde', '[CX3H1](=[OX1])[#6,*]'),
            (131, 'ketone', '[CX3](=[OX1])([#6,*])[#6,*]'),
            
            # Sulfur-containing groups
            (130, 'sulfonyl_halide', '[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]'),
            (129, 'sulfonamide', '[SX4](=[OX1])(=[OX1])[NX3]'),
            (128, 'sulfone', '[SX4](=[OX1])(=[OX1])[#6,#7,#8,#16,*]'),
            (127, 'sulfoxide', '[SX3](=[OX1])[#6,#7,#8,#16,*]'),
            (126, 'thiocyanate', '[SX2][CX2]#[NX1]'),
            
            # Other groups
            (125, 'acetal', '[CX4H]([OX2][#6,*])([OX2][#6,*])'),
            (124, 'ketal', '[CX4]([#6,*])([OX2][#6,*])[OX2][#6,*]'),
            (123, 'silicon_4_coord', '[SiX4]'),
            (122, 'boron_3_coord', '[BX3]'),
            (121, 'vinyl_halide', '[CX3]=[CX3][F,Cl,Br,I]'),
            (120, 'vinyl', '[CX3]=[CX3]'),
            (119, 'allene', '[CX3]=[CX2]=[CX3]'),
            (118, 'alcohol', '[OX2H1][#6,*]'),
            (117, 'ether', '[#6,*][OX2][#6,*]'),
            (116, 'thiol', '[SX2H1][#6,*]'),
            (115, 'thioether', '[#6,*][SX2][#6,*]'),
            (95, 'imine', '[NX2]=[CX3]'),  # Simpler pattern for N=C bond
            (89, 'nitrile', '[CX2]#[NX1]'),
            
            # Amine patterns (specific types)
            (85, 'quaternary_amine', '[NX4+]([#6,*])([#6,*])([#6,*])[#6,*]'),  # N+ with 4 carbon neighbors
            (84, 'tertiary_amine', '[NX3H0]([#6,*])([#6,*])[#6,*]'),  # N with 3 carbon neighbors
            (83, 'secondary_amine', '[NX3H1]([#6,*])[#6,*]'),  # NH with 2 carbon neighbors
            (82, 'primary_amine', '[NX3H2][#6,*]')  # NH2 with 1 carbon neighbor
        ]
        
        # Track atoms that have been matched to avoid double counting
        matched_atoms = set()
        carbonyl_in_ester_or_imide = set()
        imide_nitrogens = set()  # Track nitrogen atoms that are part of imides
        matched_carbonyl_carbons = set()  # Track carbonyl carbons that have been matched to other groups
        nitro_nitrogens = set()  # Track nitrogen atoms that are part of nitro groups
        azo_nitrogens = set()    # Track nitrogen atoms that are part of azo groups
        ring_nitrogens = set()   # Track nitrogen atoms that are part of rings
        amine_nitrogens = set()  # Track nitrogen atoms that are part of amines
        
        # First pass: identify all ring atoms and their substituents
        ring_atoms = set()
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)
            # Track ring nitrogens
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() == 'N':
                    ring_nitrogens.add(atom_idx)
        
        # First identify carbons that are part of functional groups
        functional_group_carbons = set()
        
        # Find imides first (highest priority)
        imide_pattern = Chem.MolFromSmarts('[NX3]1([#6,#1,*])[CX3](=[OX1])[#6][CX3](=[OX1])1')
        if imide_pattern:
            for match in mol.GetSubstructMatches(imide_pattern):
                # Add the nitrogen to imide_nitrogens
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'N':
                        imide_nitrogens.add(atom_idx)
                    elif atom.GetSymbol() == 'C':
                        for bond in atom.GetBonds():
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                other_atom = bond.GetOtherAtom(atom)
                                if other_atom.GetSymbol() == 'O':
                                    carbonyl_in_ester_or_imide.add(atom_idx)
                                    functional_group_carbons.add(atom_idx)
                                    matched_carbonyl_carbons.add(atom_idx)
        
        # Also try alternative imide pattern
        imide_pattern2 = Chem.MolFromSmarts('[CX3](=[OX1])[NX3][CX3](=[OX1])')
        if imide_pattern2:
            for match in mol.GetSubstructMatches(imide_pattern2):
                # Add the nitrogen to imide_nitrogens
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'N':
                        imide_nitrogens.add(atom_idx)
                    elif atom.GetSymbol() == 'C':
                        for bond in atom.GetBonds():
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                other_atom = bond.GetOtherAtom(atom)
                                if other_atom.GetSymbol() == 'O':
                                    carbonyl_in_ester_or_imide.add(atom_idx)
                                    functional_group_carbons.add(atom_idx)
                                    matched_carbonyl_carbons.add(atom_idx)
        
        # Count imides based on unique nitrogen atoms
        features['imide'] = len(imide_nitrogens)
        
        # Search for other patterns in order
        for priority, group, pattern in patterns:
            # Skip imide patterns as we've already handled them
            if group == 'imide':
                continue
            
            # Convert SMARTS to RDKit molecule pattern
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                # Find all matches
                matches = mol.GetSubstructMatches(pattern_mol)
                
                # Debug print for specific patterns
                if group in ['primary_amine', 'secondary_amine', 'tertiary_amine', 'quaternary_amine', 'ketone', 'amide', 'carboxylic_acid', 'nitro', 'azo']:
                    # print(f"\nTrying {group} pattern: {pattern}")
                    # print(f"Found {len(matches)} matches")
                    for match in matches:
                        # print(f"Match atoms: {match}")
                        pass
                
                # Count only unique matches that don't overlap with previously matched atoms
                unique_matches = set()
                for match in matches:
                    if group == 'amide':
                        # Skip if the nitrogen is part of an imide
                        is_imide = False
                        for atom_idx in match:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == 'N' and atom_idx in imide_nitrogens:
                                is_imide = True
                                break
                        if not is_imide:
                            unique_matches.add(tuple(sorted(match)))
                            # Add carbons to functional group set
                            for atom_idx in match:
                                atom = mol.GetAtomWithIdx(atom_idx)
                                if atom.GetSymbol() == 'C':
                                    functional_group_carbons.add(atom_idx)
                                    # If this is a carbonyl carbon, mark it as matched
                                    for bond in atom.GetBonds():
                                        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == 'O':
                                            matched_carbonyl_carbons.add(atom_idx)
                            matched_atoms.update(match)
                    elif group == 'nitro':
                        # For nitro groups, track the nitrogen atom
                        for atom_idx in match:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == 'N':
                                nitro_nitrogens.add(atom_idx)
                        unique_matches.add(tuple(sorted(match)))
                        matched_atoms.update(match)
                    elif group == 'azo':
                        # For azo groups, track both nitrogen atoms
                        for atom_idx in match:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == 'N':
                                azo_nitrogens.add(atom_idx)
                        unique_matches.add(tuple(sorted(match)))
                        matched_atoms.update(match)
                    elif group in ['primary_amine', 'secondary_amine', 'tertiary_amine', 'quaternary_amine']:
                        # Skip if the nitrogen is part of another functional group or ring
                        for atom_idx in match:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == 'N':
                                # Check if this nitrogen is part of an amide
                                is_amide = False
                                for bond in atom.GetBonds():
                                    other_atom = bond.GetOtherAtom(atom)
                                    if other_atom.GetSymbol() == 'C':
                                        for other_bond in other_atom.GetBonds():
                                            if other_bond.GetBondType() == Chem.BondType.DOUBLE:
                                                double_bond_atom = other_bond.GetOtherAtom(other_atom)
                                                if double_bond_atom.GetSymbol() == 'O':
                                                    is_amide = True
                                                    break
                                if not is_amide and (atom_idx not in nitro_nitrogens and 
                                    atom_idx not in azo_nitrogens and 
                                    atom_idx not in imide_nitrogens and 
                                    atom_idx not in ring_nitrogens and
                                    atom_idx not in amine_nitrogens):  # Check if already counted as an amine
                                    unique_matches.add(tuple(sorted(match)))
                                    amine_nitrogens.add(atom_idx)  # Track this nitrogen as an amine
                                    matched_atoms.update(match)
                                break
                    elif group == 'ketone':
                        # Only count ketones whose carbonyl carbon is not part of an ester, imide, or carboxylic acid
                        carbonyl_idx = match[0]  # First atom in ketone pattern is the carbonyl carbon
                        if carbonyl_idx not in carbonyl_in_ester_or_imide and carbonyl_idx not in matched_carbonyl_carbons:
                            unique_matches.add(tuple(sorted(match)))
                            functional_group_carbons.add(carbonyl_idx)
                            matched_carbonyl_carbons.add(carbonyl_idx)
                            matched_atoms.update(match)
                            break
                    elif group == 'carboxylic_acid':
                        # Mark the carbonyl carbon as matched
                        carbonyl_idx = match[0]  # First atom in carboxylic acid pattern is the carbonyl carbon
                        unique_matches.add(tuple(sorted(match)))
                        functional_group_carbons.add(carbonyl_idx)
                        matched_carbonyl_carbons.add(carbonyl_idx)
                        matched_atoms.update(match)
                    else:
                        # Skip if already matched
                        if any(atom_idx in matched_atoms for atom_idx in match):
                            continue
                        unique_matches.add(tuple(sorted(match)))
                        # Add carbons to functional group set
                        for atom_idx in match:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == 'C':
                                functional_group_carbons.add(atom_idx)
                                # If this is a carbonyl carbon, mark it as matched
                                for bond in atom.GetBonds():
                                    if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == 'O':
                                        matched_carbonyl_carbons.add(atom_idx)
                        matched_atoms.update(match)
                
                # Update the feature count
                if group == 'nitro':
                    features[group] = len(nitro_nitrogens)  # Count unique nitro groups by their nitrogen atoms
                else:
                    features[group] = len(unique_matches)
        
        # Last step: classify SP3 carbons not in rings
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                # Skip if carbon is in a ring
                if atom.IsInRing():
                    continue
                
                # Skip if carbon is not SP3
                if atom.GetHybridization() != Chem.HybridizationType.SP3:
                    continue
                
                # Count non-hydrogen neighbors
                non_h_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
                num_neighbors = len(non_h_neighbors)
                
                # Debug print for carbon classification
                # print(f"\nCarbon {atom.GetIdx()} has {num_neighbors} non-H neighbors:")
                for n in non_h_neighbors:
                    # print(f"  - Connected to {n.GetSymbol()} at index {n.GetIdx()}")
                    pass
                
                # Classify based on number of neighbors
                if num_neighbors == 1:
                    features['primary_carbon'] += 1
                    # print(f"  Classified as primary")
                elif num_neighbors == 2:
                    features['secondary_carbon'] += 1
                    # print(f"  Classified as secondary")
                elif num_neighbors == 3:
                    features['tertiary_carbon'] += 1
                    # print(f"  Classified as tertiary")
                elif num_neighbors == 4:
                    features['quaternary_carbon'] += 1
                    # print(f"  Classified as quaternary")
        
        return features

    @staticmethod
    def get_branching_factor(mol):
        """
        Calculate branching factor from SMILES structure.
        Returns average number of branches at each non-terminal carbon.
        """
        # Initialize counters
        total_branching_carbons = 0
        total_branches = 0
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and not atom.IsInRing():
                neighbors = atom.GetNeighbors()
                if len(neighbors) > 2:  # Only consider carbons with more than 2 neighbors
                    total_branching_carbons += 1
                    total_branches += len(neighbors) - 2  # Subtract 2 for the main chain
        
        return total_branches / total_branching_carbons if total_branching_carbons > 0 else 0.0

    @staticmethod
    def get_tree_depth(mol):
        """
        Calculate tree depth from SMILES structure.
        Returns the longest path from any terminal atom to any other terminal atom.
        """
        # Find terminal atoms (atoms with only one neighbor)
        terminal_atoms = [atom.GetIdx() for atom in mol.GetAtoms() 
                         if len(atom.GetNeighbors()) == 1]
        
        if len(terminal_atoms) < 2:
            return 0
        
        # Calculate all pairwise distances between terminal atoms
        max_depth = 0
        for i in range(len(terminal_atoms)):
            for j in range(i + 1, len(terminal_atoms)):
                # Get the shortest path between terminal atoms
                path = Chem.GetShortestPath(mol, terminal_atoms[i], terminal_atoms[j])
                if path:
                    # Path length is number of bonds (atoms - 1)
                    path_length = len(path) - 1
                    max_depth = max(max_depth, path_length)
        
        return max_depth

    @staticmethod
    def get_chain_features(mol):
        """Detect chain features in the molecule as a separate pass."""
        if mol is None:
            return {
                'ethyl_chain': 0,
                'propyl_chain': 0,
                'butyl_chain': 0,
                'long_chain': 0
            }
        
        # Initialize chain features
        chain_features = {
            'ethyl_chain': 0,
            'propyl_chain': 0,
            'butyl_chain': 0,
            'long_chain': 0
        }
        
        # Chain patterns in order of length (longest first)
        chain_patterns = [
            ('long_chain', '[CH2][CH2][CH2][CH2][CH2]'),  # 5 or more carbons
            ('butyl_chain', '[CH2][CH2][CH2][CH3]'),      # 4 carbons
            ('propyl_chain', '[CH2][CH2][CH3]'),          # 3 carbons
            ('ethyl_chain', '[CH2][CH3]')                 # 2 carbons
        ]
        
        # Track atoms that have been matched to chains
        chain_atoms = set()
        
        # Search for patterns in order (longest first)
        for chain_type, pattern in chain_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                matches = mol.GetSubstructMatches(pattern_mol)
                
                # Count only unique matches that don't overlap with previously matched chain atoms
                unique_matches = set()
                for match in matches:
                    # Skip if any atom in the match is already part of a chain
                    if any(atom_idx in chain_atoms for atom_idx in match):
                        continue
                    # Skip if any atom in the chain is part of a ring
                    if any(mol.GetAtomWithIdx(atom_idx).IsInRing() for atom_idx in match):
                        continue
                    unique_matches.add(tuple(sorted(match)))
                    chain_atoms.update(match)
                chain_features[chain_type] = len(unique_matches)
        return chain_features

    @classmethod
    def extract_all_features(cls, smiles):
        """Extract all features from a SMILES string for permeability prediction."""
        if not smiles:
            return None
            
        # Check if SMILES contains spaces (extended SMILES with repeat units)
        if ' ' in smiles:
            # Split by spaces and process each repeat unit separately
            repeat_units = [unit.strip() for unit in smiles.split() if unit.strip()]
            
            # Initialize combined features
            combined_features = None
            
            # Process each repeat unit and sum the features
            for repeat_unit in repeat_units:
                unit_features = cls._extract_features_from_single_smiles(repeat_unit)
                if unit_features is not None:
                    if combined_features is None:
                        combined_features = unit_features.copy()
                    else:
                        # Sum the features from each repeat unit
                        for feature, value in unit_features.items():
                            combined_features[feature] = combined_features.get(feature, 0) + value
            
            return combined_features
        else:
            # Process single SMILES as before
            return cls._extract_features_from_single_smiles(smiles)
    
    @classmethod
    def _extract_features_from_single_smiles(cls, smiles):
        """Extract features from a single SMILES string (without spaces)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Get all feature types
        hybrid_features = cls.get_hybridization_features(smiles)
        ring_features = cls.get_ring_info(mol)
        func_group_features = cls.get_functional_groups(mol)
        
        # Get tree structure features
        tree_features = {
            'branching_factor': cls.get_branching_factor(mol),
            'tree_depth': cls.get_tree_depth(mol)
        }
        # Add chain features
        tree_features.update(cls.get_chain_features(mol))
        
        # Define the exact feature order from fg_featurization.py (excluding SMILES)
        feature_order = [
            'SP_C', 'SP_N', 'SP2_C', 'SP2_N', 'SP2_O', 'SP2_S', 'SP2_B',
            'SP3_C', 'SP3_N', 'SP3_O', 'SP3_S', 'SP3_P', 'SP3_Si', 'SP3_B',
            'SP3_F', 'SP3_Cl', 'SP3_Br', 'SP3_I', 'SP3D2_S',
            'phenyls', 'cyclohexanes', 'cyclopentanes', 'cyclopentenes', 'thiophenes',
            'aromatic_rings_with_n', 'aromatic_rings_with_o', 'aromatic_rings_with_n_o',
            'aromatic_rings_with_s', 'aliphatic_rings_with_n', 'aliphatic_rings_with_o',
            'aliphatic_rings_with_n_o', 'aliphatic_rings_with_s', 'other_rings',
            'carboxylic_acid', 'anhydride', 'acyl_halide', 'carbamide', 'urea',
            'carbamate', 'thioamide', 'amide', 'ester', 'sulfonamide', 'sulfone',
            'sulfoxide', 'phosphate', 'nitro', 'acetal', 'ketal', 'isocyanate',
            'thiocyanate', 'azide', 'azo', 'imide', 'sulfonyl_halide', 'phosphonate',
            'thiourea', 'guanidine', 'silicon_4_coord', 'boron_3_coord', 'vinyl',
            'vinyl_halide', 'allene', 'alcohol', 'ether', 'aldehyde', 'ketone',
            'thiol', 'thioether', 'primary_amine', 'secondary_amine', 'tertiary_amine',
            'quaternary_amine', 'imine', 'nitrile', 'primary_carbon', 'secondary_carbon',
            'tertiary_carbon', 'quaternary_carbon',
            'branching_factor', 'tree_depth', 'ethyl_chain', 'propyl_chain', 'butyl_chain', 'long_chain'
        ]
        
        # Combine all features into a single dictionary
        all_features = {**hybrid_features, **ring_features, **func_group_features, **tree_features}
        
        # Return features in the exact order specified, with zero values for missing features
        ordered_features = {}
        for feature in feature_order:
            ordered_features[feature] = all_features.get(feature, 0)
        
        return ordered_features 