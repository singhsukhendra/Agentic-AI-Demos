
"""
Enhanced logging system for the Agentic Research Analyzer
Provides clear phase-based logging with colors and structured output
"""

import logging
import sys
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

class AgenticLogger:
    def __init__(self, name="AgenticAnalyzer", level=logging.INFO):
        self.console = Console()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create rich handler for colored output
        handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True
        )
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(message)s"
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.propagate = False
        
    def phase_start(self, phase_name, description):
        """Log the start of a major phase"""
        panel = Panel(
            f"[bold blue]{phase_name.upper()}[/bold blue]\n{description}",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.logger.info(f"🚀 Starting {phase_name} phase")
        
    def phase_end(self, phase_name, summary):
        """Log the end of a major phase"""
        self.logger.info(f"✅ Completed {phase_name} phase: {summary}")
        self.console.print()
        
    def step(self, message):
        """Log a step within a phase"""
        self.logger.info(f"  → {message}")
        
    def step_complete(self, step_name, details=""):
        """Log the completion of a step with optional details"""
        if details:
            self.logger.info(f"  ✅ {step_name}: {details}")
        else:
            self.logger.info(f"  ✅ {step_name}")
        
    def result(self, key, value):
        """Log a key result"""
        self.logger.info(f"  📊 {key}: [bold green]{value}[/bold green]")
        
    def warning(self, message):
        """Log a warning"""
        self.logger.warning(f"⚠️  {message}")
        
    def error(self, message):
        """Log an error"""
        self.logger.error(f"❌ {message}")
        
    def info(self, message):
        """Log general information"""
        self.logger.info(message)
        
    def cycle_start(self):
        """Log the start of a complete sense-plan-act cycle"""
        title = Text("AGENTIC AI RESEARCH PAPER ANALYZER", style="bold magenta")
        subtitle = Text("Demonstrating Sense-Plan-Act Cycle", style="italic")
        
        panel = Panel(
            f"{title}\n{subtitle}\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
        
    def cycle_end(self, total_time):
        """Log the end of a complete cycle"""
        panel = Panel(
            f"[bold green]ANALYSIS COMPLETE[/bold green]\n\nTotal execution time: {total_time:.2f} seconds",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
