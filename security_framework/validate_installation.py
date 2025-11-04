#!/usr/bin/env python3
"""
Validate that the Security Framework is properly installed and dependencies are available.
"""

import sys


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"  ❌ Python 3.8+ required. Found: {sys.version}")
        return False
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    dependencies = {
        'dnspython': 'dns',
        'python-nmap': 'nmap',
    }
    
    all_ok = True
    for package, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package} installed")
        except ImportError:
            print(f"  ❌ {package} not installed. Install with: pip install {package}")
            all_ok = False
    
    return all_ok


def check_nmap_executable():
    """Check if Nmap executable is available."""
    print("\nChecking Nmap executable...")
    import shutil
    
    nmap_path = shutil.which('nmap')
    if nmap_path:
        print(f"  ✅ Nmap found at: {nmap_path}")
        
        # Try to get version
        try:
            import subprocess
            result = subprocess.run(['nmap', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            version_line = result.stdout.split('\n')[0]
            print(f"     {version_line}")
        except Exception:
            pass
        
        return True
    else:
        print("  ⚠️  Nmap not found in PATH")
        print("     Install Nmap: sudo apt-get install nmap (Linux)")
        print("                    brew install nmap (macOS)")
        print("                    Download from https://nmap.org (Windows)")
        return False


def check_framework_imports():
    """Check if framework modules can be imported."""
    print("\nChecking framework imports...")
    
    try:
        from security_framework import SecurityFramework, DNSRecon, FirewallTester
        print("  ✅ Core framework imports successful")
        
        # Try to instantiate
        framework = SecurityFramework()
        print("  ✅ SecurityFramework instantiation successful")
        
        dns = DNSRecon(framework)
        print("  ✅ DNSRecon instantiation successful")
        
        fw = FirewallTester(framework)
        print("  ✅ FirewallTester instantiation successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import/instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Security Framework - Installation Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Nmap Executable", check_nmap_executable),
        ("Framework Imports", check_framework_imports),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ Error during {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed! Framework is ready to use.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
