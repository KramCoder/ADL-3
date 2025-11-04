"""
Command-line interface for the Security Framework.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core import SecurityFramework
from .dns_recon import DNSRecon
from .firewall_testing import FirewallTester


def dns_recon_command(args):
    """Execute DNS reconnaissance commands."""
    framework = SecurityFramework(output_dir=args.output, verbose=args.verbose)
    dns_recon = DNSRecon(framework)
    
    if args.command == 'query':
        results = dns_recon.query_dns_records(args.domain, args.record_types)
        print("\nDNS Records:")
        for record_type, values in results.items():
            if values:
                print(f"  {record_type}: {', '.join(values)}")
    
    elif args.command == 'subdomain':
        if args.wordlist:
            subdomains = dns_recon.subdomain_enumeration_wordlist(
                args.domain, args.wordlist, args.workers
            )
        else:
            subdomains = dns_recon.subdomain_bruteforce(
                args.domain, max_workers=args.workers
            )
        
        print(f"\nFound {len(subdomains)} subdomains:")
        for subdomain in sorted(subdomains):
            print(f"  {subdomain}")
    
    elif args.command == 'zone':
        results = dns_recon.dns_zone_transfer(args.domain)
        if results['success']:
            print(f"\nZone transfer successful! Found {len(results['records'])} records:")
            for record in results['records'][:20]:  # Show first 20
                print(f"  {record['name']} {record['type']} {record['data']}")
        else:
            print("\nZone transfer failed or not allowed.")
    
    elif args.command == 'full':
        result = dns_recon.full_dns_recon(
            args.domain,
            enable_zone_transfer=args.zone_transfer,
            enable_subdomain_bruteforce=args.subdomain_enum,
            wordlist=None if not args.wordlist else None  # Would load from file
        )
        print(f"\nDNS reconnaissance completed. Results saved to {framework.output_dir}")
        print(f"Found {len(result.results.get('subdomains', []))} subdomains")


def firewall_test_command(args):
    """Execute firewall testing commands."""
    framework = SecurityFramework(output_dir=args.output, verbose=args.verbose)
    firewall_tester = FirewallTester(framework)
    
    if args.command == 'scan':
        if args.scan_type == 'tcp':
            results = firewall_tester.port_scan_tcp(args.target, args.ports, args.timeout)
            print(f"\nTCP Port Scan Results for {args.target}:")
            open_ports = [p for p, s in results.items() if s == 'open']
            if open_ports:
                print(f"  Open ports: {', '.join(map(str, sorted(open_ports)))}")
            else:
                print("  No open ports found")
        
        elif args.scan_type == 'udp':
            results = firewall_tester.port_scan_udp(args.target, args.ports, args.timeout)
            print(f"\nUDP Port Scan Results for {args.target}:")
            open_ports = [p for p, s in results.items() if s == 'open']
            if open_ports:
                print(f"  Open ports: {', '.join(map(str, sorted(open_ports)))}")
            else:
                print("  No open ports found")
        
        elif args.scan_type == 'nmap':
            results = firewall_tester.nmap_scan(
                args.target, 
                scan_type=args.nmap_scan_type,
                ports=args.ports,
                arguments=args.nmap_args
            )
            print(f"\nNmap Scan Results for {args.target}:")
            if 'hosts' in results:
                for host, info in results['hosts'].items():
                    print(f"  Host: {host} ({info.get('hostname', 'N/A')})")
                    print(f"  State: {info.get('state', 'N/A')}")
                    if info.get('ports'):
                        print("  Open ports:")
                        for port, port_info in info['ports'].items():
                            if port_info['state'] == 'open':
                                print(f"    {port}/{protocol}: {port_info.get('name', 'unknown')} "
                                      f"({port_info.get('product', '')} {port_info.get('version', '')})")
    
    elif args.command == 'detect':
        results = firewall_tester.firewall_detection(args.target)
        print(f"\nFirewall Detection Results for {args.target}:")
        print(f"  Firewall detected: {results['firewall_detected']}")
        if results['filtered_ports']:
            print(f"  Filtered ports: {', '.join(map(str, results['filtered_ports'][:20]))}")
        if results['open_ports']:
            print(f"  Open ports: {', '.join(map(str, results['open_ports'][:20]))}")
    
    elif args.command == 'evade':
        results = firewall_tester.firewall_evasion_techniques(args.target, args.ports)
        print(f"\nFirewall Evasion Test Results for {args.target}:")
        for technique, result in results.items():
            if result:
                print(f"  {technique}: {len(result.get('open_ports', []))} open ports found")
    
    elif args.command == 'full':
        result = firewall_tester.full_firewall_test(
            args.target,
            ports=args.ports,
            enable_evasion=args.evasion
        )
        print(f"\nFirewall testing completed. Results saved to {framework.output_dir}")
        open_tcp = len([p for p, s in result.results.get('port_scan_tcp', {}).items() if s == 'open'])
        print(f"  Found {open_tcp} open TCP ports")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Security Testing Framework - Firewall Testing and DNS Reconnaissance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-o', '--output', default='./scan_results',
                       help='Output directory for scan results (default: ./scan_results)')
    
    subparsers = parser.add_subparsers(dest='module', help='Module to use')
    
    # DNS Recon subparser
    dns_parser = subparsers.add_parser('dns', help='DNS reconnaissance module')
    dns_subparsers = dns_parser.add_subparsers(dest='command', help='DNS command')
    
    # DNS query
    dns_query = dns_subparsers.add_parser('query', help='Query DNS records')
    dns_query.add_argument('domain', help='Domain to query')
    dns_query.add_argument('-r', '--record-types', nargs='+',
                          default=['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME'],
                          help='Record types to query')
    
    # DNS subdomain
    dns_subdomain = dns_subparsers.add_parser('subdomain', help='Enumerate subdomains')
    dns_subdomain.add_argument('domain', help='Domain to enumerate')
    dns_subdomain.add_argument('-w', '--wordlist', help='Path to wordlist file')
    dns_subdomain.add_argument('--workers', type=int, default=50,
                              help='Number of concurrent workers')
    
    # DNS zone transfer
    dns_zone = dns_subparsers.add_parser('zone', help='Attempt DNS zone transfer')
    dns_zone.add_argument('domain', help='Domain for zone transfer')
    
    # DNS full recon
    dns_full = dns_subparsers.add_parser('full', help='Full DNS reconnaissance')
    dns_full.add_argument('domain', help='Domain to recon')
    dns_full.add_argument('--zone-transfer', action='store_true', default=True,
                         help='Attempt zone transfer')
    dns_full.add_argument('--subdomain-enum', action='store_true', default=True,
                         help='Enable subdomain enumeration')
    dns_full.add_argument('-w', '--wordlist', help='Path to wordlist file')
    
    # Firewall Testing subparser
    fw_parser = subparsers.add_parser('firewall', help='Firewall testing module')
    fw_subparsers = fw_parser.add_subparsers(dest='command', help='Firewall command')
    
    # Firewall scan
    fw_scan = fw_subparsers.add_parser('scan', help='Port scanning')
    fw_scan.add_argument('target', help='Target host or IP')
    fw_scan.add_argument('-t', '--scan-type', choices=['tcp', 'udp', 'nmap'],
                        default='tcp', help='Type of scan')
    fw_scan.add_argument('-p', '--ports', default='1-1000',
                       help='Port range or list (default: 1-1000)')
    fw_scan.add_argument('--timeout', type=float, default=1.0,
                       help='Connection timeout')
    fw_scan.add_argument('--nmap-scan-type', default='syn',
                        choices=['syn', 'tcp', 'udp', 'stealth'],
                        help='Nmap scan type')
    fw_scan.add_argument('--nmap-args', default='-sV',
                        help='Additional nmap arguments')
    
    # Firewall detect
    fw_detect = fw_subparsers.add_parser('detect', help='Detect firewall')
    fw_detect.add_argument('target', help='Target host or IP')
    
    # Firewall evade
    fw_evade = fw_subparsers.add_parser('evade', help='Test evasion techniques')
    fw_evade.add_argument('target', help='Target host or IP')
    fw_evade.add_argument('-p', '--ports', default='1-1000',
                         help='Port range to test')
    
    # Firewall full test
    fw_full = fw_subparsers.add_parser('full', help='Full firewall test')
    fw_full.add_argument('target', help='Target host or IP')
    fw_full.add_argument('-p', '--ports', default='1-1000',
                        help='Port range to scan')
    fw_full.add_argument('--evasion', action='store_true', default=True,
                        help='Enable evasion techniques')
    
    args = parser.parse_args()
    
    if not args.module:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.module == 'dns':
            dns_recon_command(args)
        elif args.module == 'firewall':
            firewall_test_command(args)
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
