# Security Testing Framework

A comprehensive Python framework for **firewall testing** and **DNS reconnaissance**. This framework integrates various security testing tools to provide a unified interface for penetration testing and security assessments.

## Features

### DNS Reconnaissance
- **DNS Record Queries**: Query A, AAAA, MX, NS, TXT, SOA, CNAME records
- **Subdomain Enumeration**: Brute force and dictionary-based subdomain discovery
- **DNS Zone Transfer**: Attempt AXFR zone transfers
- **Reverse DNS Lookup**: IP to hostname resolution
- **Comprehensive DNS Recon**: Full automated DNS reconnaissance

### Firewall Testing
- **Port Scanning**: TCP and UDP port scanning
- **Nmap Integration**: Full Nmap scanner integration with multiple scan types
- **Firewall Detection**: Identify firewall presence and characteristics
- **Evasion Techniques**: Test various firewall bypass methods:
  - Fragmentation
  - Timing manipulation
  - Source port spoofing
  - Decoy scans
- **Port Knocking**: Test port knocking sequences
- **Comprehensive Testing**: Full automated firewall assessment

## Installation

### Prerequisites

- Python 3.8+
- Nmap (must be installed separately)
- Root/Administrator privileges (for some advanced scanning features)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Nmap

**Linux:**
```bash
sudo apt-get install nmap  # Debian/Ubuntu
sudo yum install nmap      # CentOS/RHEL
```

**macOS:**
```bash
brew install nmap
```

**Windows:**
Download from: https://nmap.org/download.html

## Usage

### Command-Line Interface

#### DNS Reconnaissance

**Query DNS records:**
```bash
python -m security_framework.cli dns query example.com
```

**Enumerate subdomains:**
```bash
python -m security_framework.cli dns subdomain example.com
```

**With custom wordlist:**
```bash
python -m security_framework.cli dns subdomain example.com -w /path/to/wordlist.txt
```

**Attempt DNS zone transfer:**
```bash
python -m security_framework.cli dns zone example.com
```

**Full DNS reconnaissance:**
```bash
python -m security_framework.cli dns full example.com --zone-transfer --subdomain-enum
```

#### Firewall Testing

**TCP port scan:**
```bash
python -m security_framework.cli firewall scan 192.168.1.1 -t tcp -p 1-1000
```

**UDP port scan:**
```bash
python -m security_framework.cli firewall scan 192.168.1.1 -t udp
```

**Nmap SYN scan:**
```bash
python -m security_framework.cli firewall scan 192.168.1.1 -t nmap --nmap-scan-type syn
```

**Detect firewall:**
```bash
python -m security_framework.cli firewall detect 192.168.1.1
```

**Test evasion techniques:**
```bash
python -m security_framework.cli firewall evade 192.168.1.1 -p 1-1000
```

**Full firewall test:**
```bash
python -m security_framework.cli firewall full 192.168.1.1 --evasion
```

### Python API

#### DNS Reconnaissance

```python
from security_framework import SecurityFramework, DNSRecon

# Initialize framework
framework = SecurityFramework(output_dir="./results")

# Create DNS recon instance
dns_recon = DNSRecon(framework)

# Query DNS records
records = dns_recon.query_dns_records("example.com")
print(records)

# Enumerate subdomains
subdomains = dns_recon.subdomain_bruteforce("example.com")
print(f"Found {len(subdomains)} subdomains")

# Full reconnaissance
result = dns_recon.full_dns_recon(
    "example.com",
    enable_zone_transfer=True,
    enable_subdomain_bruteforce=True
)
```

#### Firewall Testing

```python
from security_framework import SecurityFramework, FirewallTester

# Initialize framework
framework = SecurityFramework(output_dir="./results")

# Create firewall tester instance
firewall_tester = FirewallTester(framework)

# TCP port scan
tcp_results = firewall_tester.port_scan_tcp("192.168.1.1", ports=[22, 80, 443, 8080])
print(tcp_results)

# Nmap scan
nmap_results = firewall_tester.nmap_scan("192.168.1.1", scan_type="syn", ports="1-1000")
print(nmap_results)

# Firewall detection
firewall_info = firewall_tester.firewall_detection("192.168.1.1")
print(f"Firewall detected: {firewall_info['firewall_detected']}")

# Full firewall test
result = firewall_tester.full_firewall_test(
    "192.168.1.1",
    ports="1-1000",
    enable_evasion=True
)
```

## Output

All scan results are automatically saved to JSON files in the output directory (default: `./scan_results`). Results include:

- Target information
- Timestamp
- Scan type
- Detailed findings
- Raw tool output (where applicable)

Results can be exported in JSON or text format for further analysis.

## Tools Integrated

### DNS Reconnaissance
- **dnspython**: DNS protocol library
- **socket**: Native Python DNS resolution
- **Custom subdomain enumeration**: Multi-threaded brute force

### Firewall Testing
- **Nmap**: Network mapper and port scanner
- **python-nmap**: Python Nmap library
- **socket**: Native Python port scanning
- **Custom evasion techniques**: Multiple bypass methods

## Security and Legal Notice

⚠️ **IMPORTANT**: This framework is designed for authorized security testing only.

- Only use this tool on systems you own or have explicit written permission to test
- Unauthorized scanning and testing is illegal in most jurisdictions
- Always obtain proper authorization before conducting security assessments
- Follow responsible disclosure practices for discovered vulnerabilities
- The authors are not responsible for misuse of this tool

## Limitations

- Some advanced features require root/administrator privileges
- Firewall evasion techniques may be detected by modern security systems
- DNS zone transfers are rarely successful due to modern security configurations
- Subdomain enumeration speed depends on network conditions and target response times
- Some tools (like Nmap) must be installed separately

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is provided for educational and authorized security testing purposes only.

## Acknowledgments

This framework integrates and builds upon various open-source security tools:
- Nmap (https://nmap.org)
- dnspython (https://www.dnspython.org)
- Various security testing methodologies and techniques

## Support

For issues, questions, or contributions, please open an issue on the project repository.
