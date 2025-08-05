#!/usr/bin/env python3
"""
Systematic Grade Impact Analyzer
Analyzes the 5% impact of each grade on WVTR, TS, and EAB properties.

Strategy:
1. Test each grade in binary (50/50), ternary (33/33/33), and quaternary (25/25/25/25) blends
2. Create baseline and 5% perturbed versions for each blend
3. Measure impact on WVTR, TS, and EAB properties
4. Aggregate results to show average impact of each grade

Usage:
    python grade_impact_analyzer.py
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from itertools import combinations
import random
from typing import Dict, List, Tuple, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.input_parser import load_and_validate_material_dictionary, parse_polymer_input
from modules.prediction_engine import predict_single_property
from modules.prediction_utils import PROPERTY_CONFIGS

class GradeImpactAnalyzer:
    def __init__(self):
        """Initialize the grade impact analyzer."""
        self.material_dict = load_and_validate_material_dictionary()
        if self.material_dict is None:
            raise ValueError("Failed to load material dictionary")
        
        # Load available materials
        self.materials_df = pd.read_csv('material-smiles-dictionary.csv')
        self.grades = self._get_available_grades()
        
        # Properties to analyze
        self.properties = ['wvtr', 'ts', 'eab', 'cobb']
        
        # Results storage
        self.results = {
            'binary': {},
            'ternary': {},
            'quaternary': {}
        }
        
        # Environmental parameters
        self.env_params = {
            'Temperature (°C)': 38,
            'Relative Humidity (%)': 90,
            'Thickness (um)': 100
        }
    
    def _get_available_grades(self) -> List[Tuple[str, str]]:
        """Get list of available material-grade combinations."""
        grades = []
        for _, row in self.materials_df.iterrows():
            grades.append((row['Material'].strip(), row['Grade'].strip()))
        return grades
    
    def _create_blend_string(self, polymers: List[Tuple[str, str, float]]) -> str:
        """Create blend string for input parsing."""
        blend_parts = []
        for material, grade, concentration in polymers:
            blend_parts.extend([material, grade, str(concentration)])
        return ", ".join(blend_parts)
    
    def _predict_properties(self, blend_string: str) -> Dict[str, float]:
        """Predict properties for a given blend string using modules directly."""
        results = {}
        
        try:
            # Parse the blend string to get polymers using the input parser
            polymers, parsed_env_params = parse_polymer_input(blend_string, 'wvtr')
            if polymers is None:
                print(f"Failed to parse blend string: {blend_string}")
                return {prop: None for prop in self.properties}
            
            # Merge environmental parameters
            available_env_params = self.env_params.copy()
            if parsed_env_params:
                available_env_params.update(parsed_env_params)
            
            # Predict each property
            for prop in self.properties:
                try:
                    result = predict_single_property(
                        prop, 
                        polymers, 
                        available_env_params, 
                        self.material_dict,
                        include_errors=False
                    )
                    if result and 'prediction' in result:
                        results[prop] = result['prediction']
                    else:
                        results[prop] = None
                except Exception as e:
                    print(f"Error predicting {prop}: {e}")
                    results[prop] = None
            
        except Exception as e:
            print(f"Error in _predict_properties: {e}")
            return {prop: None for prop in self.properties}
        
        return results
    
    def _create_baseline_blend(self, blend_type: str, selected_grades: List[Tuple[str, str]], target_grade: Tuple[str, str]) -> List[Tuple[str, str, float]]:
        """Create baseline blend with 0% of target grade and equal distribution among others."""
        baseline_blend = []
        
        # Find the target grade and other grades
        other_grades = []
        for material, grade in selected_grades:
            if (material, grade) == target_grade:
                # Target grade gets 0% concentration
                baseline_blend.append((material, grade, 0.0))
            else:
                other_grades.append((material, grade))
        
        # Distribute remaining 100% equally among other grades
        if len(other_grades) > 0:
            concentration = 1.0 / len(other_grades)
            for material, grade in other_grades:
                baseline_blend.append((material, grade, concentration))
        
        return baseline_blend
    
    def _create_perturbed_blend(self, baseline_blend: List[Tuple[str, str, float]], 
                               target_grade: Tuple[str, str], perturbation: float = 0.05) -> List[Tuple[str, str, float]]:
        """Create perturbed blend with 5% of target grade and 95% distributed among others."""
        perturbed_blend = []
        target_found = False
        
        # First pass: identify target and collect other concentrations
        other_concentrations = []
        
        for material, grade, concentration in baseline_blend:
            if (material, grade) == target_grade:
                target_found = True
            else:
                other_concentrations.append((material, grade, concentration))
        
        if not target_found:
            return baseline_blend
        
        # Target grade gets 5% concentration
        new_target_concentration = perturbation
        
        # Calculate remaining concentration for other grades (95%)
        remaining_concentration = 1.0 - new_target_concentration
        
        # Scale other concentrations proportionally
        total_other = sum(conc for _, _, conc in other_concentrations)
        if total_other > 0:
            scale_factor = remaining_concentration / total_other
            
            # Build perturbed blend
            for material, grade, concentration in baseline_blend:
                if (material, grade) == target_grade:
                    perturbed_blend.append((material, grade, new_target_concentration))
                else:
                    new_concentration = concentration * scale_factor
                    perturbed_blend.append((material, grade, new_concentration))
        else:
            # If no other concentrations, just return baseline
            return baseline_blend
        
        return perturbed_blend
    
    def _calculate_impact(self, baseline_results: Dict[str, float], 
                         perturbed_results: Dict[str, float]) -> Dict[str, float]:
        """Calculate percentage impact of perturbation."""
        impact = {}
        for prop in self.properties:
            if (baseline_results.get(prop) is not None and 
                perturbed_results.get(prop) is not None and 
                baseline_results[prop] != 0):
                impact[prop] = ((perturbed_results[prop] - baseline_results[prop]) / baseline_results[prop]) * 100
            else:
                impact[prop] = None
        return impact
    
    def analyze_binary_blends(self):
        """Analyze all binary combinations."""
        print("Analyzing binary blends...")
        
        # Test all binary combinations
        binary_combinations = list(combinations(self.grades, 2))
        print(f"Testing {len(binary_combinations)} binary combinations...")
        
        for i, (grade1, grade2) in enumerate(binary_combinations):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(binary_combinations)}")
            
            # Test impact of each grade
            for target_grade in [grade1, grade2]:
                # Create baseline blend with 0% of target grade
                baseline_blend = self._create_baseline_blend('binary', [grade1, grade2], target_grade)
                baseline_string = self._create_blend_string(baseline_blend)
                baseline_results = self._predict_properties(baseline_string)
                
                # Create perturbed blend with 5% of target grade
                perturbed_blend = self._create_perturbed_blend(baseline_blend, target_grade)
                perturbed_string = self._create_blend_string(perturbed_blend)
                perturbed_results = self._predict_properties(perturbed_string)
                
                impact = self._calculate_impact(baseline_results, perturbed_results)
                
                # Store results
                grade_key = f"{target_grade[0]} {target_grade[1]}"
                if grade_key not in self.results['binary']:
                    self.results['binary'][grade_key] = []
                self.results['binary'][grade_key].append(impact)
    
    def analyze_ternary_blends(self, num_samples_per_grade: int = 2):
        """Analyze ternary combinations by iterating through each grade."""
        print(f"Analyzing ternary combinations - {num_samples_per_grade} combinations per grade...")
        
        total_combinations = 0
        
        # Iterate through each grade
        for target_grade in self.grades:
            print(f"Testing grade: {target_grade[0]} {target_grade[1]}")
            
            # Get other grades (excluding the target grade)
            other_grades = [g for g in self.grades if g != target_grade]
            
            # Create random combinations of other grades (no target grade)
            for _ in range(num_samples_per_grade):
                if len(other_grades) >= 2:
                    # Randomly select 2 other grades (no target grade)
                    selected_others = random.sample(other_grades, 2)
                    grade1, grade2 = selected_others[0], selected_others[1]
                    
                    # Test impact of adding target grade to this blend
                    baseline_blend = self._create_baseline_blend('ternary', [grade1, grade2, target_grade], target_grade)
                    baseline_string = self._create_blend_string(baseline_blend)
                    baseline_results = self._predict_properties(baseline_string)
                    
                    perturbed_blend = self._create_perturbed_blend(baseline_blend, target_grade)
                    perturbed_string = self._create_blend_string(perturbed_blend)
                    perturbed_results = self._predict_properties(perturbed_string)
                    
                    impact = self._calculate_impact(baseline_results, perturbed_results)
                    
                    # Store results
                    grade_key = f"{target_grade[0]} {target_grade[1]}"
                    if grade_key not in self.results['ternary']:
                        self.results['ternary'][grade_key] = []
                    self.results['ternary'][grade_key].append(impact)
                    
                    total_combinations += 1
        
        print(f"Completed {total_combinations} ternary combinations")
    
    def analyze_quaternary_blends(self, num_samples_per_grade: int = 1):
        """Analyze quaternary combinations by iterating through each grade."""
        print(f"Analyzing quaternary combinations - {num_samples_per_grade} combinations per grade...")
        
        total_combinations = 0
        
        # Iterate through each grade
        for target_grade in self.grades:
            print(f"Testing grade: {target_grade[0]} {target_grade[1]}")
            
            # Get other grades (excluding the target grade)
            other_grades = [g for g in self.grades if g != target_grade]
            
            # Create random combinations of other grades (no target grade)
            for _ in range(num_samples_per_grade):
                if len(other_grades) >= 3:
                    # Randomly select 3 other grades (no target grade)
                    selected_others = random.sample(other_grades, 3)
                    grade1, grade2, grade3 = selected_others[0], selected_others[1], selected_others[2]
                    
                    # Test impact of adding target grade to this blend
                    baseline_blend = self._create_baseline_blend('quaternary', [grade1, grade2, grade3, target_grade], target_grade)
                    baseline_string = self._create_blend_string(baseline_blend)
                    baseline_results = self._predict_properties(baseline_string)
                    
                    perturbed_blend = self._create_perturbed_blend(baseline_blend, target_grade)
                    perturbed_string = self._create_blend_string(perturbed_blend)
                    perturbed_results = self._predict_properties(perturbed_string)
                    
                    impact = self._calculate_impact(baseline_results, perturbed_results)
                    
                    # Store results
                    grade_key = f"{target_grade[0]} {target_grade[1]}"
                    if grade_key not in self.results['quaternary']:
                        self.results['quaternary'][grade_key] = []
                    self.results['quaternary'][grade_key].append(impact)
                    
                    total_combinations += 1
        
        print(f"Completed {total_combinations} quaternary combinations")
    
    def calculate_average_impacts(self) -> Dict[str, Dict[str, float]]:
        """Calculate average impact for each grade across all blend types."""
        print("Calculating average impacts...")
        
        # Combine all results for each grade
        all_impacts = {}
        
        for blend_type, grade_results in self.results.items():
            for grade, impacts in grade_results.items():
                if grade not in all_impacts:
                    all_impacts[grade] = []
                all_impacts[grade].extend(impacts)
        
        # Calculate averages
        average_impacts = {}
        for grade, impacts in all_impacts.items():
            if impacts:
                # Filter out None values
                valid_impacts = {prop: [] for prop in self.properties}
                for impact in impacts:
                    for prop in self.properties:
                        if impact.get(prop) is not None:
                            valid_impacts[prop].append(impact[prop])
                
                # Calculate averages
                avg_impact = {}
                for prop in self.properties:
                    if valid_impacts[prop]:
                        avg_impact[prop] = np.mean(valid_impacts[prop])
                    else:
                        avg_impact[prop] = None
                
                average_impacts[grade] = avg_impact
        
        return average_impacts
    
    def save_results(self, average_impacts: Dict[str, Dict[str, float]], thickness: int):
        """Save results to CSV and JSON files."""
        print("Saving results...")
        
        # Save detailed results
        with open('grade_impact_detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save average impacts
        with open('grade_impact_averages.json', 'w') as f:
            json.dump(average_impacts, f, indent=2, default=str)
        
        # Create CSV summary for all blend types combined
        rows = []
        for grade, impacts in average_impacts.items():
            row = {
                'Grade': grade,
                'WVTR_Impact_%': impacts.get('wvtr'),
                'TS_Impact_%': impacts.get('ts'),
                'EAB_Impact_%': impacts.get('eab'),
                'Cobb_Impact_%': impacts.get('cobb')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f'grade_impact_summary_{thickness}um.csv', index=False)
        
        # Create separate CSV files for each blend type
        for blend_type in ['binary', 'ternary', 'quaternary']:
            blend_impacts = self.calculate_blend_type_averages(blend_type)
            if blend_impacts:
                rows = []
                for grade, impacts in blend_impacts.items():
                    row = {
                        'Grade': grade,
                        'WVTR_Impact_%': impacts.get('wvtr'),
                        'TS_Impact_%': impacts.get('ts'),
                        'EAB_Impact_%': impacts.get('eab'),
                        'Cobb_Impact_%': impacts.get('cobb')
                    }
                    rows.append(row)
                
                df_blend = pd.DataFrame(rows)
                df_blend.to_csv(f'grade_impact_{blend_type}_{thickness}um.csv', index=False)
        
        print("Results saved to:")
        print("- grade_impact_detailed_results.json")
        print("- grade_impact_averages.json") 
        print(f"- grade_impact_summary_{thickness}um.csv")
        print(f"- grade_impact_binary_{thickness}um.csv")
        print(f"- grade_impact_ternary_{thickness}um.csv")
        print(f"- grade_impact_quaternary_{thickness}um.csv")
    
    def calculate_blend_type_averages(self, blend_type: str) -> Dict[str, Dict[str, float]]:
        """Calculate average impact for each grade within a specific blend type."""
        if blend_type not in self.results:
            return {}
        
        grade_results = self.results[blend_type]
        average_impacts = {}
        
        for grade, impacts in grade_results.items():
            if impacts:
                # Filter out None values
                valid_impacts = {prop: [] for prop in self.properties}
                for impact in impacts:
                    for prop in self.properties:
                        if impact.get(prop) is not None:
                            valid_impacts[prop].append(impact[prop])
                
                # Calculate averages
                avg_impact = {}
                for prop in self.properties:
                    if valid_impacts[prop]:
                        avg_impact[prop] = np.mean(valid_impacts[prop])
                    else:
                        avg_impact[prop] = None
                
                average_impacts[grade] = avg_impact
        
        return average_impacts
    
    def run_analysis(self, test_mode: bool = False):
        """Run the complete grade impact analysis for multiple thicknesses."""
        print("=" * 60)
        print("SYSTEMATIC GRADE IMPACT ANALYZER")
        print("=" * 60)
        print(f"Available grades: {len(self.grades)}")
        print(f"Properties to analyze: {self.properties}")
        print(f"Temperature: {self.env_params['Temperature (°C)']}°C")
        print(f"Relative Humidity: {self.env_params['Relative Humidity (%)']}%")
        print(f"Test mode: {test_mode}")
        print()
        
        if test_mode:
            # Use fewer grades for testing
            self.grades = self.grades[:5]  # Only first 5 grades
            print(f"Test mode: Using only {len(self.grades)} grades")
        
        # Thicknesses to analyze
        thicknesses = [10, 50, 250]
        
        for thickness in thicknesses:
            print(f"\n{'='*50}")
            print(f"ANALYZING THICKNESS: {thickness} μm")
            print(f"{'='*50}")
            
            # Update thickness for this analysis
            self.env_params['Thickness (um)'] = thickness
            
            # Reset results for this thickness
            self.results = {
                'binary': {},
                'ternary': {},
                'quaternary': {}
            }
            
            # Run analysis for each blend type
            self.analyze_binary_blends()
            print()
            
            ternary_samples_per_grade = 5 if test_mode else 50
            self.analyze_ternary_blends(num_samples_per_grade=ternary_samples_per_grade)
            print()
            
            quaternary_samples_per_grade = 2 if test_mode else 25
            self.analyze_quaternary_blends(num_samples_per_grade=quaternary_samples_per_grade)
            print()
            
            # Calculate and save results for this thickness
            average_impacts = self.calculate_average_impacts()
            self.save_results(average_impacts, thickness)
            
            # Print summary for this thickness
            print(f"\n{'='*50}")
            print(f"ANALYSIS COMPLETE FOR {thickness} μm")
            print(f"{'='*50}")
            print("Top 5 grades by average impact:")
            
            # Calculate overall impact score
            grade_scores = []
            for grade, impacts in average_impacts.items():
                valid_impacts = [v for v in impacts.values() if v is not None]
                if valid_impacts:
                    avg_score = np.mean([abs(v) for v in valid_impacts])
                    grade_scores.append((grade, avg_score))
            
            grade_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (grade, score) in enumerate(grade_scores[:5]):
                print(f"{i+1}. {grade}: {score:.2f}% average impact")
        
        print(f"\n{'='*60}")
        print("ALL THICKNESS ANALYSES COMPLETE")
        print(f"{'='*60}")
        print("Files generated:")
        for thickness in thicknesses:
            print(f"- grade_impact_summary_{thickness}um.csv")
        print("- grade_impact_detailed_results.json")
        print("- grade_impact_averages.json")

def main():
    """Main function."""
    try:
        # Check for test mode
        test_mode = '--test' in sys.argv
        if test_mode:
            sys.argv.remove('--test')
        
        analyzer = GradeImpactAnalyzer()
        analyzer.run_analysis(test_mode=test_mode)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 