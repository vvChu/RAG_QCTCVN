from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

@dataclass
class FileReport:
    """Report for a single file."""
    file_name: str
    status: str # "success", "failed", "skipped"
    chunks_indexed: int = 0
    coverage: float = 0.0
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class IngestionReport:
    """Aggregated report for an ingestion run."""
    run_id: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = None
    files: List[FileReport] = field(default_factory=list)
    
    @property
    def total_files(self) -> int:
        return len(self.files)
    
    @property
    def success_count(self) -> int:
        return sum(1 for f in self.files if f.status == "success")
        
    @property
    def failure_count(self) -> int:
        return sum(1 for f in self.files if f.status == "failed")
        
    @property
    def total_chunks(self) -> int:
        return sum(f.chunks_indexed for f in self.files)
        
    def add_result(self, context: Any):
        """Add result from PipelineContext."""
        status = "success" if not context.errors else "failed"
        if not context.metadata and not context.errors:
             status = "skipped"
             
        report = FileReport(
            file_name=Path(context.file_path).name,
            status=status,
            chunks_indexed=context.stats.get('chunks_indexed', 0),
            coverage=context.stats.get('coverage', 0.0),
            errors=context.errors,
            stats=context.stats
        )
        self.files.append(report)
        
    def save(self, output_path: str):
        """Save report to JSON."""
        data = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "summary": {
                "total_files": self.total_files,
                "success": self.success_count,
                "failed": self.failure_count,
                "total_chunks": self.total_chunks
            },
            "details": [
                {
                    "file": f.file_name,
                    "status": f.status,
                    "chunks": f.chunks_indexed,
                    "coverage": f.coverage,
                    "errors": f.errors
                }
                for f in self.files
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
