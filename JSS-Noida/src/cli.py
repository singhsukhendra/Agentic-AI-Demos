
"""
Command Line Interface for the Agentic Research Paper Analyzer
Provides easy-to-use CLI for demonstrating the sense-plan-act cycle
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from .logger import AgenticLogger
from .sense import SenseAgent
from .plan import PlanAgent
from .act import ActAgent
from .ai_pipeline import create_ai_pipeline, AIAnalysisResult

class ResearchAnalyzerCLI:
    """Main CLI application class"""
    
    def __init__(self, mode='rule', config_path=None):
        self.mode = mode
        self.config_path = config_path
        self.logger = AgenticLogger()
        
        if mode == 'ai':
            # Initialize AI pipeline
            try:
                self.ai_sense_agent, self.ai_plan_agent, self.ai_act_agent, self.ai_logger = create_ai_pipeline(config_path)
                self.logger.info("AI pipeline initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize AI pipeline: {str(e)}")
                self.logger.info("Falling back to rule-based mode")
                self.mode = 'rule'
        
        if mode == 'rule' or self.mode == 'rule':
            # Initialize rule-based agents
            self.sense_agent = SenseAgent(self.logger)
            self.plan_agent = PlanAgent(self.logger)
            self.act_agent = ActAgent(self.logger)
    
    def run(self, args):
        """Main execution method"""
        start_time = time.time()
        
        try:
            # Start the agentic cycle
            self.logger.cycle_start()
            
            # Get paper text
            paper_text = self._get_paper_text(args.paper, args.stdin)
            if not paper_text:
                self.logger.error("No paper text provided")
                return 1
            
            # Execute the Sense-Plan-Act cycle
            self._execute_cycle(paper_text, args)
            
            # End cycle
            total_time = time.time() - start_time
            self.logger.cycle_end(total_time)
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.error("Analysis interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def _get_paper_text(self, paper_path: Optional[str], use_stdin: bool) -> str:
        """Get paper text from file or stdin"""
        if use_stdin:
            self.logger.info("Reading paper text from stdin...")
            return sys.stdin.read().strip()
        
        if paper_path:
            paper_file = Path(paper_path)
            if not paper_file.exists():
                self.logger.error(f"Paper file not found: {paper_path}")
                return ""
            
            self.logger.info(f"Reading paper from: {paper_path}")
            try:
                return paper_file.read_text(encoding='utf-8')
            except Exception as e:
                self.logger.error(f"Error reading file: {str(e)}")
                return ""
        
        self.logger.error("No input source specified. Use --paper <file> or --stdin")
        return ""
    
    def _execute_cycle(self, paper_text: str, args):
        """Execute the complete sense-plan-act cycle"""
        
        if self.mode == 'ai':
            # AI-powered cycle
            self.logger.info("Executing AI-powered analysis cycle")
            
            # SENSE Phase
            metadata = self.ai_sense_agent.sense(paper_text)
            
            # PLAN Phase  
            plan = self.ai_plan_agent.plan(metadata)
            
            # ACT Phase
            analysis = self.ai_act_agent.act(paper_text, metadata, plan)
            
            # Output results with AI-specific formatting
            self._output_ai_results(analysis, args)
            
        else:
            # Rule-based cycle
            self.logger.info("Executing rule-based analysis cycle")
            
            # SENSE Phase
            metadata = self.sense_agent.sense(paper_text)
            
            # PLAN Phase  
            plan = self.plan_agent.plan(metadata)
            
            # ACT Phase
            analysis = self.act_agent.act(paper_text, metadata, plan)
            
            # Output results
            self._output_results(analysis, args)
    
    def _output_results(self, analysis, args):
        """Output analysis results in requested format"""
        if args.output_format == 'json':
            self._output_json(analysis, args.output)
        elif args.output_format == 'summary':
            self._output_summary(analysis)
        else:  # detailed
            self._output_detailed(analysis)
    
    def _output_json(self, analysis, output_file: Optional[str]):
        """Output results as JSON"""
        # Convert analysis to JSON-serializable format
        result_data = {
            'metadata': analysis.metadata_summary,
            'execution_summary': analysis.execution_summary,
            'overall_assessment': analysis.overall_assessment,
            'quality_score': analysis.quality_score,
            'recommendations': analysis.recommendations,
            'task_results': [
                {
                    'task_name': r.task_name,
                    'status': r.status,
                    'execution_time': r.execution_time,
                    'findings': r.findings,
                    'confidence_score': r.confidence_score,
                    'recommendations': r.recommendations
                }
                for r in analysis.task_results
            ]
        }
        
        json_output = json.dumps(result_data, indent=2, default=str)
        
        if output_file:
            Path(output_file).write_text(json_output)
            self.logger.info(f"Results saved to: {output_file}")
        else:
            print(json_output)
    
    def _output_summary(self, analysis):
        """Output concise summary"""
        
        # Summary panel
        summary_text = f"""
[bold]Paper Type:[/bold] {analysis.metadata_summary['paper_type']}
[bold]Research Domain:[/bold] {analysis.metadata_summary['research_domain']}
[bold]Quality Score:[/bold] {analysis.quality_score:.1f}/10
[bold]Word Count:[/bold] {analysis.metadata_summary['word_count']:,}
[bold]Complexity:[/bold] {analysis.metadata_summary['complexity_score']:.1f}/10
        """.strip()
        
        panel = Panel(summary_text, title="Analysis Summary", border_style="green")
        self.logger.console.print(panel)
        
        # Key findings
        if analysis.overall_assessment.get('strengths'):
            self.logger.console.print("\n[bold green]Strengths:[/bold green]")
            for strength in analysis.overall_assessment['strengths']:
                self.logger.console.print(f"  ✓ {strength}")
        
        if analysis.overall_assessment.get('weaknesses'):
            self.logger.console.print("\n[bold red]Areas for Improvement:[/bold red]")
            for weakness in analysis.overall_assessment['weaknesses']:
                self.logger.console.print(f"  ⚠ {weakness}")
        
        # Top recommendations
        if analysis.recommendations:
            self.logger.console.print("\n[bold blue]Top Recommendations:[/bold blue]")
            for i, rec in enumerate(analysis.recommendations, 1):
                self.logger.console.print(f"  {i}. {rec}")
    
    def _output_detailed(self, analysis):
        """Output detailed analysis results"""
        
        # Metadata table
        metadata_table = Table(title="Paper Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="white")
        
        metadata_table.add_row("Title", analysis.metadata_summary.get('title', 'N/A'))
        metadata_table.add_row("Authors", str(len(analysis.metadata_summary.get('authors', []))))
        metadata_table.add_row("Paper Type", analysis.metadata_summary.get('paper_type', 'N/A'))
        metadata_table.add_row("Research Domain", analysis.metadata_summary.get('research_domain', 'N/A'))
        metadata_table.add_row("Word Count", f"{analysis.metadata_summary.get('word_count', 0):,}")
        metadata_table.add_row("Sections", str(len(analysis.metadata_summary.get('sections', []))))
        metadata_table.add_row("Keywords", str(len(analysis.metadata_summary.get('keywords', []))))
        metadata_table.add_row("Complexity Score", f"{analysis.metadata_summary.get('complexity_score', 0):.1f}/10")
        
        self.logger.console.print(metadata_table)
        
        # Task results table
        task_table = Table(title="Task Execution Results")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Status", style="white")
        task_table.add_column("Time (s)", style="yellow")
        task_table.add_column("Confidence", style="green")
        
        for result in analysis.task_results:
            status_color = "green" if result.status == 'completed' else "red"
            task_table.add_row(
                result.task_name,
                f"[{status_color}]{result.status}[/{status_color}]",
                f"{result.execution_time:.2f}",
                f"{result.confidence_score:.2f}"
            )
        
        self.logger.console.print(task_table)
        
        # Overall assessment
        assessment_text = f"""
[bold]Overall Quality:[/bold] {analysis.overall_assessment.get('overall_quality', 'unknown')}
[bold]Success Rate:[/bold] {analysis.overall_assessment.get('success_rate', 0):.1%}
[bold]Average Confidence:[/bold] {analysis.overall_assessment.get('average_confidence', 0):.2f}
[bold]Quality Score:[/bold] {analysis.quality_score:.1f}/10
        """.strip()
        
        assessment_panel = Panel(assessment_text, title="Overall Assessment", border_style="blue")
        self.logger.console.print(assessment_panel)
        
        # Detailed findings for each task
        if any(result.findings for result in analysis.task_results):
            self.logger.console.print("\n[bold]Detailed Task Findings:[/bold]")
            for result in analysis.task_results:
                if result.findings and result.status == 'completed':
                    self.logger.console.print(f"\n[cyan]{result.task_name}:[/cyan]")
                    for key, value in result.findings.items():
                        self.logger.console.print(f"  • {key}: {value}")
        
        # All recommendations
        if analysis.recommendations:
            self.logger.console.print(f"\n[bold blue]All Recommendations ({len(analysis.recommendations)}):[/bold blue]")
            for i, rec in enumerate(analysis.recommendations, 1):
                self.logger.console.print(f"  {i}. {rec}")
    
    def _output_ai_results(self, analysis: AIAnalysisResult, args):
        """Output AI analysis results with LLM reasoning"""
        if args.output_format == 'json':
            self._output_ai_json(analysis, args.output)
        elif args.output_format == 'summary':
            self._output_ai_summary(analysis, args)
        else:  # detailed
            self._output_ai_detailed(analysis, args)
    
    def _output_ai_json(self, analysis: AIAnalysisResult, output_file: Optional[str]):
        """Output AI results as JSON with LLM reasoning"""
        result_data = {
            'mode': 'ai',
            'metadata': analysis.metadata_summary,
            'execution_summary': analysis.execution_summary,
            'overall_assessment': analysis.overall_assessment,
            'quality_score': analysis.quality_score,
            'recommendations': analysis.recommendations,
            'task_results': analysis.task_results,
            'llm_reasoning': analysis.llm_reasoning,
            'total_tokens_used': analysis.total_tokens_used,
            'total_cost_estimate': analysis.total_cost_estimate
        }
        
        json_output = json.dumps(result_data, indent=2, default=str)
        
        if output_file:
            Path(output_file).write_text(json_output)
            self.logger.info(f"AI analysis results saved to: {output_file}")
        else:
            print(json_output)
    
    def _output_ai_summary(self, analysis: AIAnalysisResult, args=None):
        """Output AI analysis summary with key insights"""
        
        # Summary panel with AI-specific info
        summary_text = f"""
[bold]Analysis Mode:[/bold] AI-Powered
[bold]Paper Type:[/bold] {analysis.metadata_summary.get('paper_type', 'Unknown')}
[bold]Research Domain:[/bold] {analysis.metadata_summary.get('research_domain', 'Unknown')}
[bold]Quality Score:[/bold] {analysis.quality_score:.1f}/10
[bold]Word Count:[/bold] {analysis.metadata_summary.get('word_count', 0):,}
[bold]Complexity:[/bold] {analysis.metadata_summary.get('complexity_score', 0):.1f}/10
[bold]Tokens Used:[/bold] {analysis.total_tokens_used:,}
[bold]Est. Cost:[/bold] ${analysis.total_cost_estimate:.4f}
        """.strip()
        
        panel = Panel(summary_text, title="AI Analysis Summary", border_style="blue")
        self.logger.console.print(panel)
        
        # Key findings from AI
        if analysis.overall_assessment.get('strengths'):
            self.logger.console.print("\n[bold green]AI-Identified Strengths:[/bold green]")
            for strength in analysis.overall_assessment['strengths']:
                self.logger.console.print(f"  ✓ {strength}")
        
        if analysis.overall_assessment.get('weaknesses'):
            self.logger.console.print("\n[bold red]AI-Identified Areas for Improvement:[/bold red]")
            for weakness in analysis.overall_assessment['weaknesses']:
                self.logger.console.print(f"  ⚠ {weakness}")
        
        # AI recommendations
        if analysis.recommendations:
            self.logger.console.print("\n[bold blue]AI Recommendations:[/bold blue]")
            for i, rec in enumerate(analysis.recommendations, 1):
                self.logger.console.print(f"  {i}. {rec}")
        
        # Show brief AI reasoning if available
        if hasattr(args, 'show_reasoning') and args.show_reasoning and analysis.llm_reasoning:
            self.logger.console.print("\n[bold cyan]AI Reasoning Summary:[/bold cyan]")
            for phase, reasoning in analysis.llm_reasoning.items():
                self.logger.console.print(f"  [cyan]{phase}:[/cyan] {reasoning}")
    
    def _output_ai_detailed(self, analysis: AIAnalysisResult, args=None):
        """Output detailed AI analysis results"""
        
        # AI Analysis header
        self.logger.console.print("[bold blue]🤖 AI-Powered Research Paper Analysis[/bold blue]\n")
        
        # Metadata table with AI insights
        metadata_table = Table(title="Paper Metadata (AI-Extracted)")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="white")
        
        metadata_table.add_row("Title", analysis.metadata_summary.get('title', 'N/A'))
        metadata_table.add_row("Paper Type", analysis.metadata_summary.get('paper_type', 'N/A'))
        metadata_table.add_row("Research Domain", analysis.metadata_summary.get('research_domain', 'N/A'))
        metadata_table.add_row("Word Count", f"{analysis.metadata_summary.get('word_count', 0):,}")
        metadata_table.add_row("Complexity Score", f"{analysis.metadata_summary.get('complexity_score', 0):.1f}/10")
        metadata_table.add_row("Key Contributions", str(len(analysis.metadata_summary.get('key_contributions', []))))
        metadata_table.add_row("Novelty Assessment", analysis.metadata_summary.get('novelty_assessment', 'N/A'))
        metadata_table.add_row("Technical Depth", analysis.metadata_summary.get('technical_depth', 'N/A'))
        
        self.logger.console.print(metadata_table)
        
        # AI Task results
        task_table = Table(title="AI Task Execution Results")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Status", style="white")
        task_table.add_column("Time (s)", style="yellow")
        task_table.add_column("Confidence", style="green")
        task_table.add_column("LLM Provider", style="magenta")
        
        for result in analysis.task_results:
            status_color = "green" if result.get('status') == 'completed' else "red"
            task_table.add_row(
                result.get('task_name', 'Unknown'),
                f"[{status_color}]{result.get('status', 'unknown')}[/{status_color}]",
                f"{result.get('execution_time', 0):.2f}",
                f"{result.get('confidence_score', 0):.2f}",
                result.get('llm_provider', 'N/A')
            )
        
        self.logger.console.print(task_table)
        
        # AI Overall assessment
        assessment_text = f"""
[bold]AI Overall Quality:[/bold] {analysis.overall_assessment.get('overall_quality', 'unknown')}
[bold]Success Rate:[/bold] {analysis.execution_summary.get('success_rate', 0):.1%}
[bold]Average Confidence:[/bold] {analysis.execution_summary.get('average_confidence', 0):.2f}
[bold]Quality Score:[/bold] {analysis.quality_score:.1f}/10
[bold]Total Tokens Used:[/bold] {analysis.total_tokens_used:,}
[bold]Estimated Cost:[/bold] ${analysis.total_cost_estimate:.4f}
        """.strip()
        
        assessment_panel = Panel(assessment_text, title="AI Overall Assessment", border_style="blue")
        self.logger.console.print(assessment_panel)
        
        # Detailed AI findings
        if any(result.get('findings') for result in analysis.task_results):
            self.logger.console.print("\n[bold]Detailed AI Task Findings:[/bold]")
            for result in analysis.task_results:
                if result.get('findings') and result.get('status') == 'completed':
                    self.logger.console.print(f"\n[cyan]{result.get('task_name', 'Unknown')}:[/cyan]")
                    findings = result.get('findings', {})
                    for key, value in findings.items():  # Show all findings
                        self.logger.console.print(f"  • {key}: {str(value)}")
        
        # AI Recommendations
        if analysis.recommendations:
            self.logger.console.print(f"\n[bold blue]AI-Generated Recommendations ({len(analysis.recommendations)}):[/bold blue]")
            for i, rec in enumerate(analysis.recommendations, 1):
                self.logger.console.print(f"  {i}. {rec}")
        
        # LLM Reasoning (if requested)
        if hasattr(args, 'show_reasoning') and args.show_reasoning and analysis.llm_reasoning:
            self.logger.console.print(f"\n[bold magenta]LLM Reasoning Process:[/bold magenta]")
            for phase, reasoning in analysis.llm_reasoning.items():
                if reasoning:
                    self.logger.console.print(f"\n[magenta]{phase.upper()}:[/magenta]")
                    self.logger.console.print(f"  {reasoning}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Agentic Research Paper Analyzer - Demonstrating Sense-Plan-Act AI Cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AI-powered analysis (default)
  %(prog)s --mode ai --paper sample_data/paper1.txt
  %(prog)s --mode ai --paper my_paper.txt --output-format summary
  %(prog)s --mode ai --stdin --output-format json --output results.json
  
  # Rule-based analysis (legacy)
  %(prog)s --mode rule --paper sample_data/paper1.txt
  
  # Compare modes
  %(prog)s --mode rule --paper sample_data/paper1.txt --output-format summary
  %(prog)s --mode ai --paper sample_data/paper1.txt --output-format summary
  
  # AI-specific options
  %(prog)s --mode ai --paper sample_data/paper1.txt --show-reasoning
  %(prog)s --mode ai --paper sample_data/paper1.txt --provider openai
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', '-m',
        choices=['ai', 'rule'],
        default='ai',
        help='Analysis mode: ai (LLM-powered) or rule (rule-based) - default: ai'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--paper', '-p',
        type=str,
        help='Path to research paper text file'
    )
    input_group.add_argument(
        '--stdin',
        action='store_true',
        help='Read paper text from standard input'
    )
    
    # Output options
    parser.add_argument(
        '--output-format', '-f',
        choices=['detailed', 'summary', 'json'],
        default='detailed',
        help='Output format (default: detailed)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file (default: stdout)'
    )
    
    # AI-specific options
    parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'local'],
        help='Preferred LLM provider for AI mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: use environment variables)'
    )
    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Show LLM reasoning process in output (AI mode only)'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show token usage and cost statistics (AI mode only)'
    )
    
    # Debug options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed error traces'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='AI-Powered Agentic Research Analyzer 2.0.0'
    )
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = ResearchAnalyzerCLI(mode=args.mode, config_path=args.config)
    exit_code = cli.run(args)
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
