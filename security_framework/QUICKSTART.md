# Quick Start Guide

## Installation

1. **Install Nmap** (required for firewall testing):
   ```bash
   # Linux
   sudo apt-get install nmap
   
   # macOS
   brew install nmap
   
   # Windows
   # Download from https://nmap.org/download.html
   ```

2. **Install Python dependencies**:
   ```bash
   cd security_framework
   pip install -r requirements.txt
   ```

3. **Optional: Install as package**:
   ```bash
   pip install -e .
   ```

## Quick Examples

### DNS Reconnaissance

**Query DNS records:**
```bash
python -m security_framework.cli dns query google.com
```

**Find subdomains:**
```bash
python -m security_framework.cli dns subdomain example.com
```

**Full DNS recon:**
```bash
python -m security_framework.cli dns full example.com
```

### Firewall Testing

**Scan common ports:**
```bash
python -m security_framework.cli firewall scan 192.168.1.1 -t tcp -p 22,80,443
```

**Detect firewall:**
```bash
python -m security_framework.cli firewall detect 192.168.1.1
```

**Full firewall test:**
```bash
python -m security_framework.cli firewall full 192.168.1.1
```

## Python API Example

```python
from security_framework import SecurityFramework, DNSRecon, FirewallTester

# Initialize
framework = SecurityFramework(output_dir="./results")

# DNS Recon
dns = DNSRecon(framework)
subdomains = dns.subdomain_bruteforce("example.com")
print(f"Found {len(subdomains)} subdomains")

# Firewall Test
fw = FirewallTester(framework)
ports = fw.port_scan_tcp("192.168.1.1", ports=[22, 80, 443])
print(f"Open ports: {[p for p, s in ports.items() if s == 'open']}")
```

## Important Notes

⚠️ **Always get authorization before testing!**

- Only test systems you own or have permission to test
- Unauthorized scanning is illegal
- Use responsibly and ethically

## Output

Results are saved to `./scan_results/` by default. Use `-o` to specify a different directory.

```bash
python -m security_framework.cli dns query example.com -o ./my_results
```

## Troubleshooting

**"Nmap not found" error:**
- Make sure Nmap is installed and in your PATH
- On Linux, you may need: `sudo apt-get install nmap`

**Permission denied errors:**
- Some advanced scans require root/administrator privileges
- Use `sudo` on Linux/macOS for advanced features

**DNS queries timing out:**
- Check your network connection
- Some DNS servers may rate-limit queries
- Try increasing timeout values in the code

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/basic_usage.py](examples/basic_usage.py) for more examples
- Review the CLI help: `python -m security_framework.cli --help`
