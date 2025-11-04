"""
Core framework classes and utilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ScanResult:
    """Base class for scan results."""
    target: str
    timestamp: str
    scan_type: str
    results: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class SecurityFramework:
    """Main framework class for coordinating security tests."""
    
    def __init__(self, output_dir: str = "./scan_results", verbose: bool = False):
        """
        Initialize the security framework.
        
        Args:
            output_dir: Directory to save scan results
            verbose: Enable verbose logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.results: List[ScanResult] = []
    
    def save_result(self, result: ScanResult) -> Path:
        """
        Save a scan result to file.
        
        Args:
            result: ScanResult object to save
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.scan_type}_{result.target}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        return filepath
    
    def get_results(self, scan_type: Optional[str] = None) -> List[ScanResult]:
        """
        Get stored results, optionally filtered by scan type.
        
        Args:
            scan_type: Optional filter for scan type
            
        Returns:
            List of ScanResult objects
        """
        if scan_type:
            return [r for r in self.results if r.scan_type == scan_type]
        return self.results
    
    def export_all_results(self, format: str = "json") -> Path:
        """
        Export all results to a single file.
        
        Args:
            format: Export format (json, txt)
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"all_results_{timestamp}.json"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2)
        else:
            filename = f"all_results_{timestamp}.txt"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                for result in self.results:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Scan Type: {result.scan_type}\n")
                    f.write(f"Target: {result.target}\n")
                    f.write(f"Timestamp: {result.timestamp}\n")
                    f.write(f"{'='*60}\n")
                    f.write(json.dumps(result.results, indent=2))
                    f.write("\n")
        
        self.logger.info(f"All results exported to {filepath}")
        return filepath
