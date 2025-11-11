#!/usr/bin/env python3
"""
Domain Extensions Setup - Corporación Favorita Time Series Forecasting
========================================================================
Installs domain-specific packages for time series analysis and forecasting.

Run after setup_base_environment.py to add:
- Advanced forecasting models (Prophet, LightGBM)
- Machine learning frameworks (scikit-learn, TensorFlow)
- Time series specific tools
- Enhanced visualization libraries

Usage:
    python setup_domain_extensions.py
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(status, message):
    """Print status message with indicator."""
    indicators = {
        'OK': '✓',
        'WARNING': '⚠',
        'ERROR': '✗',
        'INFO': 'ℹ'
    }
    print(f"{indicators.get(status, '•')} {status}: {message}")


def run_command(command, description):
    """Execute shell command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def check_package_installed(package_name):
    """Check if package is already installed."""
    try:
        __import__(package_name.split('[')[0].replace('-', '_'))
        return True
    except ImportError:
        return False


def install_domain_packages():
    """Install domain-specific packages for time series forecasting."""
    
    print_header("Time Series Forecasting - Domain Extensions Setup")
    
    # Define domain-specific packages (not in base requirements)
    # Format: (pip_name, import_name, description)
    packages = [
    # Machine Learning & Preprocessing
    ("scikit-learn==1.3.2", "sklearn", "Machine learning algorithms and preprocessing"),
    
    # Advanced Time Series Models
    ("prophet==1.1.5", "prophet", "Facebook Prophet for time series forecasting"),
    ("pmdarima==2.0.4", "pmdarima", "Auto-ARIMA for automated model selection"),
    
    # Gradient Boosting
    ("lightgbm==4.1.0", "lightgbm", "LightGBM for tree-based forecasting"),
    ("xgboost==2.0.3", "xgboost", "XGBoost for gradient boosting models"),
    
    # Deep Learning (optional - large install)
    ("tensorflow==2.15.0", "tensorflow", "TensorFlow for LSTM/neural networks"),
    
    # Enhanced Visualization
    ("plotly==5.18.0", "plotly", "Interactive visualizations"),
    
    # Time Series Utilities
    ("tsfresh==0.20.2", "tsfresh", "Automated feature extraction for time series"),
    ("sktime==0.25.0", "sktime", "Scikit-learn compatible time series toolkit"),
    ("tslearn==0.6.2", "tslearn", "Time series machine learning algorithms"),
]
    
    installed = []
    skipped = []
    failed = []
    
    print_status('INFO', f'Checking {len(packages)} domain-specific packages...')
    
    for pip_name, import_name, description in packages:
        package_display = pip_name.split('==')[0]
        
        # Check if already installed
        if check_package_installed(import_name):
            print_status('OK', f'{package_display} already installed - SKIPPED')
            skipped.append(package_display)
            continue
        
        # Install package
        print(f"\nInstalling {package_display}...")
        print(f"  Purpose: {description}")
        
        success, output = run_command(
            f'pip install {pip_name}',
            f'Installing {package_display}'
        )
        
        if success:
            print_status('OK', f'{package_display} installed successfully')
            installed.append(pip_name)
        else:
            print_status('ERROR', f'{package_display} installation failed')
            print(f"  Error details: {output}")
            failed.append(package_display)
    
    return installed, skipped, failed


def update_requirements_file(new_packages):
    """Update requirements.txt with newly installed packages."""
    
    print_header("Updating requirements.txt")
    
    req_file = Path('requirements.txt')
    
    if not req_file.exists():
        print_status('WARNING', 'requirements.txt not found - creating new file')
        existing_content = []
    else:
        with open(req_file, 'r') as f:
            existing_content = f.read().splitlines()
    
    # Add domain extensions section if not present
    if '# Domain-specific extensions' not in '\n'.join(existing_content):
        existing_content.append('\n# Domain-specific extensions (Time Series Forecasting)')
    
    # Add new packages
    for package in new_packages:
        if package not in '\n'.join(existing_content):
            existing_content.append(package)
    
    # Write updated requirements
    with open(req_file, 'w') as f:
        f.write('\n'.join(existing_content))
    
    print_status('OK', f'requirements.txt updated with {len(new_packages)} new packages')


def verify_installations():
    """Verify all domain packages are importable."""
    
    print_header("Verification - Domain Package Imports")
    
    test_imports = [
    ('sklearn', 'scikit-learn'),
    ('prophet', 'prophet'),
    ('pmdarima', 'pmdarima'),
    ('lightgbm', 'lightgbm'),
    ('xgboost', 'xgboost'),
    ('tensorflow', 'tensorflow'),
    ('plotly', 'plotly'),
    ('tsfresh', 'tsfresh'),
    ('sktime', 'sktime'),
    ('tslearn', 'tslearn'), 
]
    
    success_count = 0
    
    for module_name, display_name in test_imports:
        try:
            __import__(module_name)
            print_status('OK', f'{display_name} import successful')
            success_count += 1
        except ImportError as e:
            print_status('WARNING', f'{display_name} import failed: {e}')
    
    print(f"\n{success_count}/{len(test_imports)} packages verified successfully")
    
    return success_count == len(test_imports)


def print_summary(installed, skipped, failed):
    """Print installation summary."""
    
    print_header("Installation Summary")
    
    print(f"\n{'Category':<25} {'Count':<10}")
    print("-" * 35)
    print(f"{'Newly installed':<25} {len(installed):<10}")
    print(f"{'Already installed':<25} {len(skipped):<10}")
    print(f"{'Failed installations':<25} {len(failed):<10}")
    
    if installed:
        print("\nNewly installed packages:")
        for package in installed:
            print(f"  • {package}")
    
    if failed:
        print("\nWARNING - Failed installations:")
        for package in failed:
            print(f"  • {package}")
        print("\nTry installing failed packages manually:")
        for package in failed:
            print(f"  pip install {package}")


def main():
    """Main execution function."""
    
    print_header("Starting Domain Extensions Setup")
    
    # Check if running in virtual environment
    if not hasattr(sys, 'real_prefix') and not (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        print_status('WARNING', 'Not running in a virtual environment')
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Install packages
    installed, skipped, failed = install_domain_packages()
    
    # Update requirements.txt
    if installed:
        update_requirements_file(installed)
    
    # Verify installations
    all_verified = verify_installations()
    
    # Print summary
    print_summary(installed, skipped, failed)
    
    # Final status
    print_header("Setup Complete")
    
    if failed:
        print_status('WARNING', 'Some packages failed to install')
        print("Review error messages above and install manually if needed.")
    elif all_verified:
        print_status('OK', 'All domain extensions installed and verified successfully')
        print("\nYou can now use advanced time series forecasting tools:")
        print("  • Prophet for automated forecasting")
        print("  • LightGBM/XGBoost for tree-based models")
        print("  • TensorFlow for deep learning (LSTM)")
        print("  • Enhanced visualization with Plotly")
        print("  • Automated feature extraction with tsfresh")
    else:
        print_status('WARNING', 'Installation complete but some imports failed')
        print("This may be due to dependency issues. Review warnings above.")
    
    print("\nNext steps:")
    print("  1. Close and reopen your Jupyter notebook")
    print("  2. Restart the kernel to load new packages")
    print("  3. Continue with Day 1 notebook execution")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Setup failed with exception: {e}")
        sys.exit(1)
