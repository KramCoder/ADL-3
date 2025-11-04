"""
Firewall Testing Module
Handles port scanning, firewall detection, evasion techniques, and firewall bypass.
"""

import socket
import subprocess
import nmap
import ipaddress
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import ScanResult, SecurityFramework


class FirewallTester:
    """Firewall testing and port scanning tools."""
    
    def __init__(self, framework: Optional[SecurityFramework] = None):
        """
        Initialize firewall testing module.
        
        Args:
            framework: Optional SecurityFramework instance for result storage
        """
        self.framework = framework
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nm = nmap.PortScanner()
    
    def port_scan_tcp(self, target: str, ports: List[int] = None, 
                     timeout: float = 1.0) -> Dict[int, str]:
        """
        Perform TCP port scan.
        
        Args:
            target: Target host or IP
            ports: List of ports to scan (default: common ports)
            timeout: Connection timeout in seconds
            
        Returns:
            Dictionary mapping port numbers to status ('open', 'closed', 'filtered')
        """
        if ports is None:
            # Common ports
            ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 
                    993, 995, 1723, 3306, 3389, 5900, 8080, 8443]
        
        results = {}
        
        self.logger.info(f"Scanning {len(ports)} TCP ports on {target}...")
        
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            try:
                result = sock.connect_ex((target, port))
                if result == 0:
                    results[port] = 'open'
                    self.logger.info(f"Port {port} is OPEN")
                else:
                    results[port] = 'closed'
            except socket.timeout:
                results[port] = 'filtered'
            except Exception as e:
                self.logger.debug(f"Error scanning port {port}: {e}")
                results[port] = 'error'
            finally:
                sock.close()
        
        return results
    
    def port_scan_udp(self, target: str, ports: List[int] = None,
                     timeout: float = 2.0) -> Dict[int, str]:
        """
        Perform UDP port scan.
        
        Args:
            target: Target host or IP
            ports: List of ports to scan
            timeout: Connection timeout in seconds
            
        Returns:
            Dictionary mapping port numbers to status
        """
        if ports is None:
            ports = [53, 67, 68, 69, 123, 135, 137, 138, 139, 161, 162, 445, 514, 520, 631, 1434, 1900, 4500, 49152]
        
        results = {}
        
        self.logger.info(f"Scanning {len(ports)} UDP ports on {target}...")
        
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            
            try:
                # Send empty UDP packet
                sock.sendto(b'', (target, port))
                sock.recvfrom(1024)
                results[port] = 'open'
                self.logger.info(f"Port {port} is OPEN")
            except socket.timeout:
                # Timeout might mean filtered or open with no response
                results[port] = 'open|filtered'
            except Exception as e:
                self.logger.debug(f"Error scanning UDP port {port}: {e}")
                results[port] = 'closed'
            finally:
                sock.close()
        
        return results
    
    def nmap_scan(self, target: str, scan_type: str = "syn", 
                  ports: str = "1-1000", arguments: str = "-sV") -> Dict[str, Any]:
        """
        Perform Nmap scan using python-nmap.
        
        Args:
            target: Target host or IP
            scan_type: Type of scan (syn, tcp, udp, etc.)
            ports: Port range or list
            arguments: Additional nmap arguments
            
        Returns:
            Dictionary with scan results
        """
        self.logger.info(f"Running Nmap {scan_type} scan on {target}...")
        
        try:
            if scan_type == "syn":
                scan_args = f"-sS {arguments}"
            elif scan_type == "tcp":
                scan_args = f"-sT {arguments}"
            elif scan_type == "udp":
                scan_args = f"-sU {arguments}"
            elif scan_type == "stealth":
                scan_args = f"-sS -f -T2 {arguments}"  # Stealth scan with fragmentation
            else:
                scan_args = arguments
            
            self.nm.scan(target, ports, scan_args)
            
            results = {
                'command_line': self.nm.command_line(),
                'scan_info': self.nm.scaninfo(),
                'hosts': {}
            }
            
            for host in self.nm.all_hosts():
                host_info = {
                    'hostname': self.nm[host].hostname(),
                    'state': self.nm[host].state(),
                    'protocols': self.nm[host].all_protocols(),
                    'ports': {}
                }
                
                for protocol in self.nm[host].all_protocols():
                    ports_dict = self.nm[host][protocol].keys()
                    for port in ports_dict:
                        port_info = self.nm[host][protocol][port]
                        host_info['ports'][port] = {
                            'state': port_info['state'],
                            'name': port_info.get('name', ''),
                            'product': port_info.get('product', ''),
                            'version': port_info.get('version', ''),
                            'extrainfo': port_info.get('extrainfo', '')
                        }
                
                results['hosts'][host] = host_info
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during Nmap scan: {e}")
            return {'error': str(e)}
    
    def firewall_detection(self, target: str) -> Dict[str, Any]:
        """
        Detect firewall presence and characteristics.
        
        Args:
            target: Target host or IP
            
        Returns:
            Dictionary with firewall detection results
        """
        self.logger.info(f"Detecting firewall on {target}...")
        
        results = {
            'firewall_detected': False,
            'filtered_ports': [],
            'open_ports': [],
            'closed_ports': [],
            'detection_methods': {}
        }
        
        # Method 1: TCP SYN scan - filtered ports suggest firewall
        try:
            self.nm.scan(target, "1-1000", "-sS --scan-delay 100ms")
            for host in self.nm.all_hosts():
                for protocol in self.nm[host].all_protocols():
                    for port in self.nm[host][protocol].keys():
                        state = self.nm[host][protocol][port]['state']
                        if state == 'filtered':
                            results['filtered_ports'].append(port)
                            results['firewall_detected'] = True
                        elif state == 'open':
                            results['open_ports'].append(port)
                        elif state == 'closed':
                            results['closed_ports'].append(port)
            
            results['detection_methods']['tcp_syn_scan'] = {
                'filtered_count': len(results['filtered_ports']),
                'open_count': len(results['open_ports'])
            }
        except Exception as e:
            self.logger.error(f"Error in firewall detection scan: {e}")
        
        # Method 2: ACK scan - helps identify filtered ports
        try:
            self.nm.scan(target, "1-1000", "-sA")
            ack_filtered = []
            for host in self.nm.all_hosts():
                for protocol in self.nm[host].all_protocols():
                    for port in self.nm[host][protocol].keys():
                        if self.nm[host][protocol][port]['state'] == 'filtered':
                            ack_filtered.append(port)
            
            results['detection_methods']['ack_scan'] = {
                'filtered_count': len(ack_filtered)
            }
        except Exception as e:
            self.logger.debug(f"ACK scan failed: {e}")
        
        # Method 3: Check for common firewall ports
        firewall_ports = [22, 80, 443, 8080, 8443]
        detected_firewall_ports = []
        for port in firewall_ports:
            if port in results['open_ports']:
                detected_firewall_ports.append(port)
        
        results['detection_methods']['common_firewall_ports'] = detected_firewall_ports
        
        return results
    
    def firewall_evasion_techniques(self, target: str, ports: str = "1-1000") -> Dict[str, Any]:
        """
        Test various firewall evasion techniques.
        
        Args:
            target: Target host or IP
            ports: Port range to test
            
        Returns:
            Dictionary with evasion test results
        """
        self.logger.info(f"Testing firewall evasion techniques on {target}...")
        
        results = {
            'fragmentation': {},
            'timing': {},
            'source_port': {},
            'mtu_discovery': {}
        }
        
        # Technique 1: Fragmentation
        try:
            self.logger.info("Testing fragmentation evasion...")
            self.nm.scan(target, ports, "-f")  # Fragment packets
            frag_results = self._parse_nmap_results()
            results['fragmentation'] = frag_results
        except Exception as e:
            self.logger.error(f"Fragmentation test failed: {e}")
        
        # Technique 2: Timing (slow scan)
        try:
            self.logger.info("Testing slow scan evasion...")
            self.nm.scan(target, ports, "-T1")  # Paranoid timing
            timing_results = self._parse_nmap_results()
            results['timing'] = timing_results
        except Exception as e:
            self.logger.error(f"Timing test failed: {e}")
        
        # Technique 3: Source port manipulation
        try:
            self.logger.info("Testing source port evasion...")
            self.nm.scan(target, ports, "-g 53")  # Use DNS source port
            source_port_results = self._parse_nmap_results()
            results['source_port'] = source_port_results
        except Exception as e:
            self.logger.error(f"Source port test failed: {e}")
        
        # Technique 4: Decoy scan
        try:
            self.logger.info("Testing decoy scan evasion...")
            # Note: This requires proper permissions and might be detected
            # self.nm.scan(target, ports, "-D RND:10")  # Random decoys
            results['decoy'] = {'note': 'Decoy scan requires elevated privileges'}
        except Exception as e:
            self.logger.debug(f"Decoy scan not available: {e}")
        
        return results
    
    def _parse_nmap_results(self) -> Dict[str, Any]:
        """Parse Nmap scan results into a structured format."""
        results = {
            'hosts': {},
            'open_ports': [],
            'filtered_ports': [],
            'closed_ports': []
        }
        
        for host in self.nm.all_hosts():
            host_info = {
                'hostname': self.nm[host].hostname(),
                'state': self.nm[host].state(),
                'ports': {}
            }
            
            for protocol in self.nm[host].all_protocols():
                for port in self.nm[host][protocol].keys():
                    state = self.nm[host][protocol][port]['state']
                    port_info = {
                        'state': state,
                        'name': self.nm[host][protocol][port].get('name', ''),
                        'product': self.nm[host][protocol][port].get('product', '')
                    }
                    host_info['ports'][port] = port_info
                    
                    if state == 'open':
                        results['open_ports'].append(port)
                    elif state == 'filtered':
                        results['filtered_ports'].append(port)
                    elif state == 'closed':
                        results['closed_ports'].append(port)
            
            results['hosts'][host] = host_info
        
        return results
    
    def port_knocking_test(self, target: str, sequence: List[int] = None) -> Dict[str, Any]:
        """
        Test port knocking sequence.
        
        Args:
            target: Target host or IP
            sequence: Sequence of ports to knock
            
        Returns:
            Dictionary with port knocking results
        """
        if sequence is None:
            sequence = [1000, 2000, 3000]  # Default sequence
        
        self.logger.info(f"Testing port knocking sequence: {sequence}")
        
        results = {
            'sequence': sequence,
            'success': False,
            'ports_knocked': []
        }
        
        # Knock each port in sequence
        for port in sequence:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect_ex((target, port))
                sock.close()
                results['ports_knocked'].append(port)
                time.sleep(0.1)  # Small delay between knocks
            except Exception as e:
                self.logger.debug(f"Error knocking port {port}: {e}")
        
        # After knocking, test if a previously closed port is now open
        # This is a simplified test - real port knocking would require specific sequences
        test_port = 22  # Common SSH port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((target, test_port))
            if result == 0:
                results['success'] = True
                results['test_port_opened'] = test_port
            sock.close()
        except Exception as e:
            self.logger.debug(f"Error testing port after knock: {e}")
        
        return results
    
    def full_firewall_test(self, target: str, ports: str = "1-1000",
                          enable_evasion: bool = True) -> ScanResult:
        """
        Perform comprehensive firewall testing.
        
        Args:
            target: Target host or IP
            ports: Port range to scan
            enable_evasion: Enable firewall evasion techniques
            
        Returns:
            ScanResult object with all findings
        """
        self.logger.info(f"Starting comprehensive firewall test for {target}")
        
        results = {
            'target': target,
            'port_scan_tcp': {},
            'port_scan_udp': {},
            'nmap_scan': {},
            'firewall_detection': {},
            'evasion_techniques': {}
        }
        
        # TCP port scan
        self.logger.info("Performing TCP port scan...")
        tcp_ports = list(range(1, 1001))  # Convert port range
        results['port_scan_tcp'] = self.port_scan_tcp(target, tcp_ports[:100])  # Limit for demo
        
        # UDP port scan
        self.logger.info("Performing UDP port scan...")
        results['port_scan_udp'] = self.port_scan_udp(target)
        
        # Nmap scan
        self.logger.info("Performing Nmap SYN scan...")
        results['nmap_scan'] = self.nmap_scan(target, scan_type="syn", ports=ports)
        
        # Firewall detection
        self.logger.info("Detecting firewall...")
        results['firewall_detection'] = self.firewall_detection(target)
        
        # Evasion techniques
        if enable_evasion:
            self.logger.info("Testing firewall evasion techniques...")
            results['evasion_techniques'] = self.firewall_evasion_techniques(target, ports)
        
        # Create scan result
        scan_result = ScanResult(
            target=target,
            timestamp=datetime.now().isoformat(),
            scan_type="firewall_test",
            results=results
        )
        
        # Save to framework if available
        if self.framework:
            self.framework.results.append(scan_result)
            self.framework.save_result(scan_result)
        
        self.logger.info(f"Firewall testing completed for {target}")
        return scan_result
