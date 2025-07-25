
"""
PLAN Phase: Strategic analysis planning based on sensed information
Creates analysis strategy tailored to paper type and research domain
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from .sense import PaperMetadata
from .logger import AgenticLogger

@dataclass
class AnalysisTask:
    """Represents a single analysis task"""
    name: str
    description: str
    priority: int  # 1-5, where 5 is highest priority
    estimated_time: float  # in seconds
    required_data: List[str]
    output_format: str

@dataclass
class AnalysisPlan:
    """Complete analysis plan with ordered tasks"""
    strategy: str
    tasks: List[AnalysisTask]
    total_estimated_time: float
    focus_areas: List[str]
    analysis_depth: str
    expected_outputs: List[str]

class PlanAgent:
    """
    Plan Agent: Responsible for creating analysis strategies
    This agent determines what analysis to perform based on paper characteristics
    """
    
    def __init__(self, logger: AgenticLogger):
        self.logger = logger
        
        # Analysis strategies for different paper types
        self.strategies = {
            'survey': {
                'name': 'Comprehensive Survey Analysis',
                'focus': ['coverage_analysis', 'taxonomy_extraction', 'trend_identification', 'gap_analysis'],
                'depth': 'broad'
            },
            'experimental': {
                'name': 'Experimental Validation Analysis',
                'focus': ['methodology_review', 'results_analysis', 'statistical_validation', 'reproducibility_check'],
                'depth': 'deep'
            },
            'theoretical': {
                'name': 'Theoretical Foundation Analysis',
                'focus': ['mathematical_rigor', 'proof_validation', 'theoretical_contribution', 'formal_analysis'],
                'depth': 'deep'
            },
            'application': {
                'name': 'Application-Oriented Analysis',
                'focus': ['practical_utility', 'implementation_analysis', 'performance_evaluation', 'usability_assessment'],
                'depth': 'medium'
            },
            'methodology': {
                'name': 'Methodological Innovation Analysis',
                'focus': ['novelty_assessment', 'methodology_comparison', 'algorithmic_analysis', 'efficiency_evaluation'],
                'depth': 'deep'
            }
        }
        
        # Domain-specific analysis adjustments
        self.domain_adjustments = {
            'machine learning': ['model_architecture', 'training_methodology', 'performance_metrics'],
            'computer vision': ['visual_analysis', 'dataset_evaluation', 'accuracy_metrics'],
            'natural language processing': ['linguistic_analysis', 'corpus_evaluation', 'semantic_assessment'],
            'robotics': ['hardware_analysis', 'control_systems', 'real_world_applicability'],
            'cybersecurity': ['threat_analysis', 'security_evaluation', 'vulnerability_assessment'],
            'data science': ['data_quality', 'statistical_methods', 'visualization_effectiveness']
        }
    
    def plan(self, metadata: PaperMetadata) -> AnalysisPlan:
        """
        Main planning function that creates a comprehensive analysis strategy
        """
        self.logger.phase_start("PLAN", "Creating tailored analysis strategy based on paper characteristics")
        
        # Step 1: Select base strategy
        self.logger.step("Selecting analysis strategy based on paper type")
        base_strategy = self.strategies.get(metadata.paper_type, self.strategies['methodology'])
        self.logger.result("Selected strategy", base_strategy['name'])
        
        # Step 2: Adjust for research domain
        self.logger.step("Adapting strategy for research domain")
        domain_focus = self.domain_adjustments.get(metadata.research_domain, [])
        self.logger.result("Domain-specific adjustments", len(domain_focus))
        
        # Step 3: Adjust for complexity
        self.logger.step("Calibrating analysis depth based on complexity")
        analysis_depth = self._determine_analysis_depth(metadata.complexity_score)
        self.logger.result("Analysis depth", analysis_depth)
        
        # Step 4: Create task list
        self.logger.step("Generating specific analysis tasks")
        tasks = self._create_task_list(base_strategy, domain_focus, metadata)
        self.logger.result("Tasks planned", len(tasks))
        
        # Step 5: Prioritize and order tasks
        self.logger.step("Prioritizing and sequencing tasks")
        ordered_tasks = self._prioritize_tasks(tasks, metadata)
        total_time = sum(task.estimated_time for task in ordered_tasks)
        self.logger.result("Total estimated time", f"{total_time:.1f} seconds")
        
        # Step 6: Define focus areas
        self.logger.step("Identifying key focus areas")
        focus_areas = base_strategy['focus'] + domain_focus
        self.logger.result("Focus areas", len(focus_areas))
        
        # Step 7: Define expected outputs
        self.logger.step("Planning analysis outputs")
        expected_outputs = self._define_expected_outputs(metadata, ordered_tasks)
        self.logger.result("Expected outputs", len(expected_outputs))
        
        # Create final plan
        plan = AnalysisPlan(
            strategy=base_strategy['name'],
            tasks=ordered_tasks,
            total_estimated_time=total_time,
            focus_areas=focus_areas,
            analysis_depth=analysis_depth,
            expected_outputs=expected_outputs
        )
        
        self.logger.phase_end("PLAN", f"Created {len(ordered_tasks)}-task analysis plan ({total_time:.1f}s estimated)")
        return plan
    
    def _determine_analysis_depth(self, complexity_score: float) -> str:
        """Determine appropriate analysis depth based on complexity"""
        if complexity_score >= 7.0:
            return "comprehensive"
        elif complexity_score >= 4.0:
            return "detailed"
        else:
            return "standard"
    
    def _create_task_list(self, strategy: Dict, domain_focus: List[str], metadata: PaperMetadata) -> List[AnalysisTask]:
        """Create list of analysis tasks based on strategy and metadata"""
        tasks = []
        
        # Core analysis tasks based on strategy
        core_tasks = {
            'content_structure_analysis': AnalysisTask(
                name="Content Structure Analysis",
                description="Analyze document organization and logical flow",
                priority=5,
                estimated_time=2.0,
                required_data=['sections', 'word_count'],
                output_format="structured_summary"
            ),
            'keyword_significance_analysis': AnalysisTask(
                name="Keyword Significance Analysis",
                description="Evaluate importance and relevance of key terms",
                priority=4,
                estimated_time=1.5,
                required_data=['keywords', 'research_domain'],
                output_format="ranked_list"
            ),
            'research_contribution_assessment': AnalysisTask(
                name="Research Contribution Assessment",
                description="Identify and evaluate novel contributions",
                priority=5,
                estimated_time=3.0,
                required_data=['abstract', 'paper_type'],
                output_format="contribution_matrix"
            ),
            'methodology_evaluation': AnalysisTask(
                name="Methodology Evaluation",
                description="Assess research methods and approaches",
                priority=4,
                estimated_time=2.5,
                required_data=['sections', 'paper_type'],
                output_format="methodology_report"
            ),
            'quality_metrics_calculation': AnalysisTask(
                name="Quality Metrics Calculation",
                description="Calculate various quality and readability metrics",
                priority=3,
                estimated_time=1.0,
                required_data=['word_count', 'complexity_score'],
                output_format="metrics_dashboard"
            )
        }
        
        # Add core tasks
        for focus_area in strategy['focus']:
            if focus_area in ['coverage_analysis', 'methodology_review', 'mathematical_rigor', 'practical_utility', 'novelty_assessment']:
                tasks.append(core_tasks['research_contribution_assessment'])
            elif focus_area in ['taxonomy_extraction', 'results_analysis', 'proof_validation', 'implementation_analysis', 'methodology_comparison']:
                tasks.append(core_tasks['methodology_evaluation'])
        
        # Always include structure and keyword analysis
        tasks.extend([
            core_tasks['content_structure_analysis'],
            core_tasks['keyword_significance_analysis'],
            core_tasks['quality_metrics_calculation']
        ])
        
        # Add domain-specific tasks
        if 'model_architecture' in domain_focus:
            tasks.append(AnalysisTask(
                name="ML Model Architecture Analysis",
                description="Analyze machine learning model design and structure",
                priority=4,
                estimated_time=2.0,
                required_data=['keywords', 'sections'],
                output_format="architecture_summary"
            ))
        
        if 'statistical_methods' in domain_focus:
            tasks.append(AnalysisTask(
                name="Statistical Methods Review",
                description="Evaluate statistical approaches and validity",
                priority=3,
                estimated_time=1.5,
                required_data=['methodology', 'results'],
                output_format="statistical_report"
            ))
        
        # Remove duplicates
        unique_tasks = []
        seen_names = set()
        for task in tasks:
            if task.name not in seen_names:
                unique_tasks.append(task)
                seen_names.add(task.name)
        
        return unique_tasks
    
    def _prioritize_tasks(self, tasks: List[AnalysisTask], metadata: PaperMetadata) -> List[AnalysisTask]:
        """Prioritize and order tasks based on importance and dependencies"""
        # Adjust priorities based on paper characteristics
        for task in tasks:
            # Boost priority for complex papers
            if metadata.complexity_score > 6.0:
                if 'methodology' in task.name.lower() or 'contribution' in task.name.lower():
                    task.priority = min(5, task.priority + 1)
            
            # Boost priority for survey papers
            if metadata.paper_type == 'survey':
                if 'keyword' in task.name.lower() or 'structure' in task.name.lower():
                    task.priority = min(5, task.priority + 1)
        
        # Sort by priority (descending) then by estimated time (ascending)
        return sorted(tasks, key=lambda x: (-x.priority, x.estimated_time))
    
    def _define_expected_outputs(self, metadata: PaperMetadata, tasks: List[AnalysisTask]) -> List[str]:
        """Define what outputs the analysis should produce"""
        outputs = [
            "Executive Summary",
            "Research Domain Classification",
            "Paper Type Analysis",
            "Content Structure Report",
            "Key Findings Summary"
        ]
        
        # Add task-specific outputs
        for task in tasks:
            if task.output_format not in ["structured_summary", "ranked_list"]:
                outputs.append(f"{task.name} Results")
        
        # Add domain-specific outputs
        if metadata.research_domain == 'machine learning':
            outputs.append("ML Model Assessment")
        elif metadata.research_domain == 'cybersecurity':
            outputs.append("Security Analysis Report")
        
        return list(set(outputs))  # Remove duplicates
