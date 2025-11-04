"""
DNS Reconnaissance Module
Handles subdomain enumeration, DNS record queries, and DNS-based reconnaissance.
"""

import socket
import subprocess
import dns.resolver
import dns.reversename
import dns.query
import dns.zone
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import logging
import json
import concurrent.futures
from datetime import datetime

from .core import ScanResult, SecurityFramework


class DNSRecon:
    """DNS reconnaissance tools and utilities."""
    
    def __init__(self, framework: Optional[SecurityFramework] = None):
        """
        Initialize DNS reconnaissance module.
        
        Args:
            framework: Optional SecurityFramework instance for result storage
        """
        self.framework = framework
        self.logger = logging.getLogger(self.__class__.__name__)
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = 5
        self.resolver.lifetime = 10
    
    def query_dns_records(self, domain: str, record_types: List[str] = None) -> Dict[str, Any]:
        """
        Query various DNS records for a domain.
        
        Args:
            domain: Domain name to query
            record_types: List of record types (A, AAAA, MX, NS, TXT, SOA, CNAME)
            
        Returns:
            Dictionary containing DNS records
        """
        if record_types is None:
            record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME']
        
        results = {}
        
        for record_type in record_types:
            try:
                answers = self.resolver.resolve(domain, record_type)
                results[record_type] = [str(rdata) for rdata in answers]
            except dns.resolver.NoAnswer:
                results[record_type] = []
            except dns.resolver.NXDOMAIN:
                self.logger.warning(f"Domain {domain} does not exist")
                results[record_type] = []
            except Exception as e:
                self.logger.error(f"Error querying {record_type} for {domain}: {e}")
                results[record_type] = []
        
        return results
    
    def reverse_dns_lookup(self, ip_address: str) -> List[str]:
        """
        Perform reverse DNS lookup.
        
        Args:
            ip_address: IP address to look up
            
        Returns:
            List of hostnames
        """
        try:
            hostname = socket.gethostbyaddr(ip_address)[0]
            return [hostname]
        except socket.herror:
            return []
        except Exception as e:
            self.logger.error(f"Error in reverse DNS lookup for {ip_address}: {e}")
            return []
    
    def subdomain_bruteforce(self, domain: str, wordlist: List[str] = None, 
                             max_workers: int = 50) -> Set[str]:
        """
        Brute force subdomains using a wordlist.
        
        Args:
            domain: Base domain
            wordlist: List of subdomain prefixes to try
            max_workers: Maximum concurrent workers
            
        Returns:
            Set of discovered subdomains
        """
        if wordlist is None:
            # Default common subdomains
            wordlist = [
                'www', 'mail', 'ftp', 'localhost', 'webmail', 'smtp', 'pop', 'ns1',
                'webdisk', 'ns2', 'cpanel', 'whm', 'autodiscover', 'autoconfig', 'm',
                'imap', 'test', 'ns', 'blog', 'pop3', 'dev', 'www2', 'admin', 'forum',
                'news', 'vpn', 'ns3', 'mail2', 'new', 'mysql', 'old', 'lists', 'support',
                'mobile', 'mx', 'static', 'docs', 'beta', 'shop', 'sql', 'secure', 'demo',
                'vpn', 'ns4', 'www3', 'api', 'cdn', 'images', 'www1', 'autodiscover',
                'm', 'imap', 'test', 'ns', 'blog', 'pop3', 'dev', 'www2', 'admin',
                'forum', 'news', 'vpn', 'ns3', 'mail2', 'new', 'mysql', 'old', 'lists'
            ]
        
        discovered = set()
        
        def check_subdomain(subdomain):
            full_domain = f"{subdomain}.{domain}"
            try:
                socket.gethostbyname(full_domain)
                discovered.add(full_domain)
                self.logger.info(f"Found subdomain: {full_domain}")
                return full_domain
            except socket.gaierror:
                return None
            except Exception as e:
                self.logger.debug(f"Error checking {full_domain}: {e}")
                return None
        
        self.logger.info(f"Brute forcing subdomains for {domain} with {len(wordlist)} entries...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_subdomain, sub) for sub in wordlist]
            concurrent.futures.wait(futures)
        
        return discovered
    
    def subdomain_enumeration_wordlist(self, domain: str, wordlist_path: str, 
                                      max_workers: int = 50) -> Set[str]:
        """
        Enumerate subdomains using a wordlist file.
        
        Args:
            domain: Base domain
            wordlist_path: Path to wordlist file
            max_workers: Maximum concurrent workers
            
        Returns:
            Set of discovered subdomains
        """
        wordlist_file = Path(wordlist_path)
        if not wordlist_file.exists():
            self.logger.error(f"Wordlist file not found: {wordlist_path}")
            return set()
        
        with open(wordlist_file, 'r') as f:
            wordlist = [line.strip() for line in f if line.strip()]
        
        return self.subdomain_bruteforce(domain, wordlist, max_workers)
    
    def dns_zone_transfer(self, domain: str) -> Dict[str, Any]:
        """
        Attempt DNS zone transfer (AXFR).
        
        Args:
            domain: Domain to attempt zone transfer for
            
        Returns:
            Dictionary with zone transfer results
        """
        results = {
            'success': False,
            'nameservers': [],
            'records': []
        }
        
        try:
            # Get nameservers
            ns_records = self.resolver.resolve(domain, 'NS')
            nameservers = [str(rdata) for rdata in ns_records]
            results['nameservers'] = nameservers
            
            # Attempt zone transfer from each nameserver
            for ns in nameservers:
                try:
                    ns_ip = socket.gethostbyname(ns.rstrip('.'))
                    zone = dns.zone.from_xfr(dns.query.xfr(ns_ip, domain))
                    
                    for name, node in zone.nodes.items():
                        for rdataset in node.rdatasets:
                            for rdata in rdataset:
                                results['records'].append({
                                    'name': str(name),
                                    'type': dns.rdatatype.to_text(rdataset.rdtype),
                                    'data': str(rdata)
                                })
                    
                    results['success'] = True
                    self.logger.info(f"Zone transfer successful from {ns}")
                    break
                except Exception as e:
                    self.logger.debug(f"Zone transfer failed from {ns}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error attempting zone transfer: {e}")
        
        return results
    
    def dns_bruteforce_records(self, domain: str, wordlist: List[str] = None) -> Dict[str, List[str]]:
        """
        Brute force DNS records (subdomains and other records).
        
        Args:
            domain: Base domain
            wordlist: List of prefixes to try
            
        Returns:
            Dictionary mapping record types to discovered records
        """
        if wordlist is None:
            wordlist = ['www', 'mail', 'ftp', 'admin', 'test', 'dev', 'api', 'cdn']
        
        results = {'A': [], 'AAAA': [], 'MX': [], 'CNAME': []}
        
        for prefix in wordlist:
            subdomain = f"{prefix}.{domain}"
            records = self.query_dns_records(subdomain, ['A', 'AAAA', 'MX', 'CNAME'])
            
            for record_type in ['A', 'AAAA', 'MX', 'CNAME']:
                if records.get(record_type):
                    results[record_type].extend(records[record_type])
        
        return results
    
    def full_dns_recon(self, domain: str, enable_zone_transfer: bool = True,
                      enable_subdomain_bruteforce: bool = True,
                      wordlist: List[str] = None) -> ScanResult:
        """
        Perform comprehensive DNS reconnaissance.
        
        Args:
            domain: Domain to recon
            enable_zone_transfer: Attempt DNS zone transfer
            enable_subdomain_bruteforce: Enable subdomain brute forcing
            wordlist: Optional wordlist for subdomain enumeration
            
        Returns:
            ScanResult object with all findings
        """
        self.logger.info(f"Starting comprehensive DNS reconnaissance for {domain}")
        
        results = {
            'domain': domain,
            'dns_records': {},
            'subdomains': [],
            'zone_transfer': {},
            'reverse_dns': {}
        }
        
        # Standard DNS queries
        self.logger.info("Querying standard DNS records...")
        results['dns_records'] = self.query_dns_records(domain)
        
        # Extract IPs for reverse DNS
        ips = []
        if 'A' in results['dns_records']:
            ips.extend(results['dns_records']['A'])
        if 'AAAA' in results['dns_records']:
            ips.extend(results['dns_records']['AAAA'])
        
        # Reverse DNS lookup
        for ip in ips:
            hostnames = self.reverse_dns_lookup(ip)
            if hostnames:
                results['reverse_dns'][ip] = hostnames
        
        # Zone transfer attempt
        if enable_zone_transfer:
            self.logger.info("Attempting DNS zone transfer...")
            results['zone_transfer'] = self.dns_zone_transfer(domain)
        
        # Subdomain enumeration
        if enable_subdomain_bruteforce:
            self.logger.info("Enumerating subdomains...")
            subdomains = self.subdomain_bruteforce(domain, wordlist)
            results['subdomains'] = sorted(list(subdomains))
            
            # Query DNS records for discovered subdomains
            for subdomain in subdomains:
                subdomain_records = self.query_dns_records(subdomain)
                results['dns_records'][subdomain] = subdomain_records
        
        # Create scan result
        scan_result = ScanResult(
            target=domain,
            timestamp=datetime.now().isoformat(),
            scan_type="dns_recon",
            results=results
        )
        
        # Save to framework if available
        if self.framework:
            self.framework.results.append(scan_result)
            self.framework.save_result(scan_result)
        
        self.logger.info(f"DNS reconnaissance completed for {domain}")
        return scan_result
