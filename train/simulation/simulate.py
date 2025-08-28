#!/usr/bin/env python3
"""
Main simulation script for polymer blend property augmentation.
Usage: python simulate.py --property <property> --number <count> [--seed <seed>]
       python simulate.py --all --number <count> [--seed <seed>]
"""

import argparse
import sys
import os
from typing import Dict, Any

# Add the simulation directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the common simulation functions
from simulation_common import run_simulation_for_property, run_all_simulations

# Import property configurations from simulation_rules
from simulation_rules import PROPERTY_CONFIGS

# Property configurations (using the restored PROPERTY_CONFIGS)
PROPERTY_RULES = PROPERTY_CONFIGS


def list_available_properties():
    """List all available properties for simulation"""
    print("Available properties for simulation:")
    print("=" * 50)
    
    for prop, config in PROPERTY_RULES.items():
        print(f"  {prop:12} - {config['name']}")
        print(f"              Default count: {config['default_count']}")
        print()
    
    print("Usage:")
    print("  python simulate.py --property <property> --number <count>")
    print("  python simulate.py --all --number <count>")
    print("  python simulate.py --list")
    print("\nExamples:")
    print("  python simulate.py --property cobb --number 10000")
    print("  python simulate.py --all --number 5000")


def validate_property(property_name: str) -> bool:
    """Validate that the specified property is supported"""
    if property_name not in PROPERTY_RULES:
        print(f"❌ Error: Property '{property_name}' is not supported.")
        print(f"Available properties: {', '.join(PROPERTY_RULES.keys())}")
        return False
    return True


def run_single_simulation(property_name: str, target_count: int, seed: int = 42) -> bool:
    """Run simulation for a single property"""
    if not validate_property(property_name):
        return False
    
    config = PROPERTY_RULES[property_name]
    
    print(f"🚀 Starting {config['name']} simulation...")
    print(f"📊 Target: {target_count:,} augmented samples")
    print(f"🎲 Random seed: {seed}")
    
    try:
        # Run the simulation using common module
        result = run_simulation_for_property(
            property_name=property_name,
            target_total=target_count,
            property_config=config
        )
        
        if result:
            print(f"\n✅ {config['name']} simulation completed successfully!")
            print(f"📁 Check the data/{property_name}/ directory for output files")
            return True
        else:
            print(f"\n❌ {config['name']} simulation failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during {config['name']} simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Polymer blend property simulation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulate.py --property cobb --number 10000
  python simulate.py --all --number 5000
  python simulate.py --property ts --number 5000 --seed 123
  python simulate.py --list
        """
    )
    
    parser.add_argument(
        '--property', '-p',
        type=str,
        help='Property to simulate (cobb, ts, wvtr, eab, compost, adhesion, otr)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run simulations for ALL properties'
    )
    
    parser.add_argument(
        '--number', '-n',
        type=int,
        help='Number of augmented samples to generate per property'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available properties'
    )
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        list_available_properties()
        return
    
    # Validate arguments
    if not args.all and not args.property:
        print("❌ Error: Must specify either --property or --all")
        parser.print_help()
        return
    
    if not args.number:
        print("❌ Error: --number is required")
        parser.print_help()
        return
    
    if args.number <= 0:
        print("❌ Error: --number must be positive")
        return
    
    # Run simulations
    if args.all:
        print("🚀 Running simulations for ALL properties...")
        success = run_all_simulations(args.number, args.seed)
        if success:
            print("\n🎉 All simulations completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Some simulations failed!")
            sys.exit(1)
    else:
        # Single property simulation
        success = run_single_simulation(args.property, args.number, args.seed)
        if success:
            print("\n🎉 Simulation completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Simulation failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
