#!/usr/bin/env python3
"""
Basic usage examples for the Security Framework.
"""

from security_framework import SecurityFramework, DNSRecon, FirewallTester


def example_dns_recon():
    """Example DNS reconnaissance."""
    print("=" * 60)
    print("DNS Reconnaissance Example")
    print("=" * 60)
    
    # Initialize framework
    framework = SecurityFramework(output_dir="./results", verbose=True)
    dns_recon = DNSRecon(framework)
    
    # Example domain (use your own or a test domain)
    domain = "example.com"
    
    # Query DNS records
    print(f"\n1. Querying DNS records for {domain}...")
    records = dns_recon.query_dns_records(domain)
    for record_type, values in records.items():
        if values:
            print(f"   {record_type}: {', '.join(values)}")
    
    # Reverse DNS lookup
    print("\n2. Reverse DNS lookup...")
    if records.get('A'):
        for ip in records['A'][:3]:  # First 3 IPs
            hostnames = dns_recon.reverse_dns_lookup(ip)
            if hostnames:
                print(f"   {ip} -> {', '.join(hostnames)}")
    
    # Subdomain enumeration (limited)
    print("\n3. Enumerating subdomains (limited)...")
    common_subs = ['www', 'mail', 'ftp', 'admin', 'test']
    subdomains = dns_recon.subdomain_bruteforce(domain, wordlist=common_subs)
    if subdomains:
        print(f"   Found subdomains: {', '.join(sorted(subdomains))}")
    else:
        print("   No subdomains found")
    
    # Full DNS recon
    print("\n4. Running full DNS reconnaissance...")
    result = dns_recon.full_dns_recon(
        domain,
        enable_zone_transfer=True,
        enable_subdomain_bruteforce=False  # Disable for faster demo
    )
    print(f"   Results saved to: {framework.output_dir}")


def example_firewall_testing():
    """Example firewall testing."""
    print("\n" + "=" * 60)
    print("Firewall Testing Example")
    print("=" * 60)
    
    # Initialize framework
    framework = SecurityFramework(output_dir="./results", verbose=True)
    firewall_tester = FirewallTester(framework)
    
    # Example target (use localhost or a test target)
    target = "127.0.0.1"
    
    # TCP port scan
    print(f"\n1. TCP port scan on {target}...")
    common_ports = [22, 23, 25, 53, 80, 110, 443, 3306, 8080]
    tcp_results = firewall_tester.port_scan_tcp(target, ports=common_ports)
    open_ports = [p for p, s in tcp_results.items() if s == 'open']
    if open_ports:
        print(f"   Open TCP ports: {', '.join(map(str, sorted(open_ports)))}")
    else:
        print("   No open TCP ports found")
    
    # Nmap scan
    print(f"\n2. Nmap SYN scan on {target}...")
    nmap_results = firewall_tester.nmap_scan(
        target,
        scan_type="syn",
        ports="1-1000",
        arguments="-sV"
    )
    if 'hosts' in nmap_results:
        for host, info in nmap_results['hosts'].items():
            print(f"   Host: {host}")
            print(f"   State: {info.get('state', 'N/A')}")
            if info.get('ports'):
                open_count = sum(1 for p in info['ports'].values() if p['state'] == 'open')
                print(f"   Open ports: {open_count}")
    
    # Firewall detection
    print(f"\n3. Firewall detection on {target}...")
    firewall_info = firewall_tester.firewall_detection(target)
    print(f"   Firewall detected: {firewall_info['firewall_detected']}")
    if firewall_info['filtered_ports']:
        print(f"   Filtered ports: {len(firewall_info['filtered_ports'])}")
    
    # Full firewall test
    print(f"\n4. Running full firewall test on {target}...")
    result = firewall_tester.full_firewall_test(
        target,
        ports="1-1000",
        enable_evasion=False  # Disable for faster demo
    )
    print(f"   Results saved to: {framework.output_dir}")


def example_integrated_scan():
    """Example of integrated DNS and firewall scanning."""
    print("\n" + "=" * 60)
    print("Integrated Scan Example")
    print("=" * 60)
    
    framework = SecurityFramework(output_dir="./results", verbose=True)
    
    domain = "example.com"
    
    # Step 1: DNS reconnaissance to find targets
    print(f"\nStep 1: DNS reconnaissance for {domain}...")
    dns_recon = DNSRecon(framework)
    records = dns_recon.query_dns_records(domain)
    
    # Extract IP addresses
    ips = []
    if records.get('A'):
        ips.extend(records['A'])
    if records.get('AAAA'):
        ips.extend(records['AAAA'])
    
    print(f"   Found {len(ips)} IP addresses")
    
    # Step 2: Firewall testing on discovered IPs
    if ips:
        print(f"\nStep 2: Firewall testing on discovered IPs...")
        firewall_tester = FirewallTester(framework)
        
        # Test first IP (limited scan for demo)
        target_ip = ips[0]
        print(f"   Testing {target_ip}...")
        
        # Quick port scan
        common_ports = [22, 80, 443, 8080]
        tcp_results = firewall_tester.port_scan_tcp(target_ip, ports=common_ports)
        open_ports = [p for p, s in tcp_results.items() if s == 'open']
        
        if open_ports:
            print(f"   Open ports on {target_ip}: {', '.join(map(str, open_ports))}")
        else:
            print(f"   No open ports found on {target_ip}")
    
    # Export all results
    print("\nStep 3: Exporting all results...")
    export_path = framework.export_all_results(format="json")
    print(f"   All results exported to: {export_path}")


if __name__ == "__main__":
    import sys
    
    print("Security Framework - Basic Usage Examples")
    print("=" * 60)
    print("\nNote: These examples use 'example.com' and '127.0.0.1'")
    print("Replace with your own targets for real testing.")
    print("\n" + "=" * 60)
    
    try:
        # Run examples
        example_dns_recon()
        example_firewall_testing()
        example_integrated_scan()
        
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
