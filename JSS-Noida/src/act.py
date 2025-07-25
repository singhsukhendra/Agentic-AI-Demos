
"""
ACT Phase: Execute analysis plan and generate insights
Performs the actual analysis tasks and produces structured outputs
"""

import re
import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from .sense import PaperMetadata
from .plan import AnalysisPlan, AnalysisTask
from .logger import AgenticLogger

@dataclass
class AnalysisResult:
    """Represents the result of a single analysis task"""
    task_name: str
    status: str  # 'completed', 'failed', 'skipped'
    execution_time: float
    findings: Dict[str, Any]
    confidence_score: float
    recommendations: List[str]

@dataclass
class ComprehensiveAnalysis:
    """Complete analysis results"""
    metadata_summary: Dict[str, Any]
    task_results: List[AnalysisResult]
    overall_assessment: Dict[str, Any]
    recommendations: List[str]
    quality_score: float
    execution_summary: Dict[str, Any]

class ActAgent:
    """
    Act Agent: Responsible for executing the analysis plan
    This agent performs the actual analysis tasks and generates insights
    """
    
    def __init__(self, logger: AgenticLogger):
        self.logger = logger
        self.paper_text = ""
        self.metadata = None
    
    def act(self, paper_text: str, metadata: PaperMetadata, plan: AnalysisPlan) -> ComprehensiveAnalysis:
        """
        Main action function that executes the analysis plan
        """
        self.paper_text = paper_text
        self.metadata = metadata
        
        self.logger.phase_start("ACT", f"Executing {len(plan.tasks)} analysis tasks")
        
        task_results = []
        total_execution_time = 0.0
        
        # Execute each task in the plan
        for i, task in enumerate(plan.tasks, 1):
            self.logger.step(f"Executing task {i}/{len(plan.tasks)}: {task.name}")
            
            try:
                result = self._execute_task(task)
                task_results.append(result)
                total_execution_time += result.execution_time
                
                self.logger.result(f"Task {i} status", result.status)
                self.logger.result(f"Task {i} confidence", f"{result.confidence_score:.2f}")
                
            except Exception as e:
                self.logger.error(f"Task {i} failed: {str(e)}")
                failed_result = AnalysisResult(
                    task_name=task.name,
                    status='failed',
                    execution_time=0.0,
                    findings={'error': str(e)},
                    confidence_score=0.0,
                    recommendations=[]
                )
                task_results.append(failed_result)
        
        # Generate overall assessment
        self.logger.step("Generating overall assessment")
        overall_assessment = self._generate_overall_assessment(task_results)
        
        # Calculate quality score
        self.logger.step("Calculating quality metrics")
        quality_score = self._calculate_quality_score(task_results)
        self.logger.result("Overall quality score", f"{quality_score:.2f}/10")
        
        # Generate recommendations
        self.logger.step("Formulating recommendations")
        recommendations = self._generate_recommendations(task_results, overall_assessment)
        self.logger.result("Recommendations generated", len(recommendations))
        
        # Create comprehensive analysis
        analysis = ComprehensiveAnalysis(
            metadata_summary=asdict(metadata),
            task_results=task_results,
            overall_assessment=overall_assessment,
            recommendations=recommendations,
            quality_score=quality_score,
            execution_summary={
                'total_tasks': len(plan.tasks),
                'successful_tasks': len([r for r in task_results if r.status == 'completed']),
                'failed_tasks': len([r for r in task_results if r.status == 'failed']),
                'total_execution_time': total_execution_time,
                'average_confidence': statistics.mean([r.confidence_score for r in task_results if r.confidence_score > 0])
            }
        )
        
        self.logger.phase_end("ACT", f"Completed analysis with {quality_score:.1f}/10 quality score")
        return analysis
    
    def _execute_task(self, task: AnalysisTask) -> AnalysisResult:
        """Execute a single analysis task"""
        import time
        start_time = time.time()
        
        # Route to appropriate analysis method
        if "structure" in task.name.lower():
            findings = self._analyze_content_structure()
        elif "keyword" in task.name.lower():
            findings = self._analyze_keyword_significance()
        elif "contribution" in task.name.lower():
            findings = self._assess_research_contribution()
        elif "methodology" in task.name.lower():
            findings = self._evaluate_methodology()
        elif "quality" in task.name.lower():
            findings = self._calculate_quality_metrics()
        elif "ml model" in task.name.lower():
            findings = self._analyze_ml_architecture()
        elif "statistical" in task.name.lower():
            findings = self._review_statistical_methods()
        else:
            findings = self._generic_analysis(task)
        
        execution_time = time.time() - start_time
        
        # Calculate confidence based on findings quality
        confidence_score = self._calculate_task_confidence(findings)
        
        # Generate task-specific recommendations
        recommendations = self._generate_task_recommendations(task, findings)
        
        return AnalysisResult(
            task_name=task.name,
            status='completed',
            execution_time=execution_time,
            findings=findings,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
    
    def _analyze_content_structure(self) -> Dict[str, Any]:
        """Analyze the structural organization of the paper"""
        findings = {}
        
        # Section analysis
        sections = self.metadata.sections
        findings['section_count'] = len(sections)
        findings['has_standard_structure'] = any(
            section.lower() in ['introduction', 'methodology', 'results', 'conclusion']
            for section in sections
        )
        
        # Content distribution
        paragraphs = self.paper_text.split('\n\n')
        findings['paragraph_count'] = len([p for p in paragraphs if len(p.strip()) > 50])
        
        # Average paragraph length
        para_lengths = [len(p.split()) for p in paragraphs if len(p.strip()) > 50]
        findings['avg_paragraph_length'] = statistics.mean(para_lengths) if para_lengths else 0
        
        # Structural quality indicators
        findings['structure_quality'] = 'good' if findings['has_standard_structure'] and findings['section_count'] >= 4 else 'basic'
        
        return findings
    
    def _analyze_keyword_significance(self) -> Dict[str, Any]:
        """Analyze the significance and relevance of keywords"""
        findings = {}
        
        keywords = self.metadata.keywords
        findings['total_keywords'] = len(keywords)
        
        # Categorize keywords by domain relevance
        domain_keywords = []
        technical_keywords = []
        general_keywords = []
        
        technical_indicators = ['algorithm', 'model', 'system', 'method', 'analysis', 'framework']
        
        for keyword in keywords:
            if any(indicator in keyword for indicator in technical_indicators):
                technical_keywords.append(keyword)
            elif keyword in self.metadata.research_domain.split():
                domain_keywords.append(keyword)
            else:
                general_keywords.append(keyword)
        
        findings['domain_specific_keywords'] = len(domain_keywords)
        findings['technical_keywords'] = len(technical_keywords)
        findings['general_keywords'] = len(general_keywords)
        findings['keyword_diversity'] = len(set(keywords)) / len(keywords) if keywords else 0
        
        # Top keywords by frequency (simulated)
        findings['top_keywords'] = keywords[:5]
        
        return findings
    
    def _assess_research_contribution(self) -> Dict[str, Any]:
        """Assess the research contributions of the paper"""
        findings = {}
        
        text_lower = self.paper_text.lower()
        
        # Look for contribution indicators
        contribution_indicators = [
            'novel', 'new', 'innovative', 'first', 'propose', 'introduce',
            'improve', 'enhance', 'advance', 'breakthrough', 'significant'
        ]
        
        contribution_score = sum(text_lower.count(indicator) for indicator in contribution_indicators)
        findings['contribution_indicators'] = contribution_score
        
        # Novelty assessment
        novelty_terms = ['novel', 'new', 'innovative', 'original', 'unprecedented']
        novelty_score = sum(text_lower.count(term) for term in novelty_terms)
        findings['novelty_score'] = novelty_score
        
        # Impact indicators
        impact_terms = ['significant', 'important', 'substantial', 'major', 'breakthrough']
        impact_score = sum(text_lower.count(term) for term in impact_terms)
        findings['impact_score'] = impact_score
        
        # Overall contribution assessment
        total_score = contribution_score + novelty_score + impact_score
        if total_score > 10:
            findings['contribution_level'] = 'high'
        elif total_score > 5:
            findings['contribution_level'] = 'medium'
        else:
            findings['contribution_level'] = 'low'
        
        findings['contribution_summary'] = f"Paper shows {findings['contribution_level']} level contributions with {total_score} contribution indicators"
        
        return findings
    
    def _evaluate_methodology(self) -> Dict[str, Any]:
        """Evaluate the research methodology"""
        findings = {}
        
        text_lower = self.paper_text.lower()
        
        # Methodology indicators
        method_indicators = [
            'experiment', 'evaluation', 'test', 'measure', 'compare',
            'analyze', 'study', 'investigate', 'examine', 'assess'
        ]
        
        method_score = sum(text_lower.count(indicator) for indicator in method_indicators)
        findings['methodology_indicators'] = method_score
        
        # Rigor indicators
        rigor_indicators = [
            'statistical', 'significant', 'p-value', 'confidence', 'validation',
            'benchmark', 'baseline', 'control', 'systematic', 'rigorous'
        ]
        
        rigor_score = sum(text_lower.count(indicator) for indicator in rigor_indicators)
        findings['rigor_score'] = rigor_score
        
        # Reproducibility indicators
        repro_indicators = [
            'reproduce', 'replicate', 'code', 'dataset', 'implementation',
            'available', 'open source', 'github', 'parameters', 'settings'
        ]
        
        repro_score = sum(text_lower.count(indicator) for indicator in repro_indicators)
        findings['reproducibility_score'] = repro_score
        
        # Overall methodology quality
        total_method_score = method_score + rigor_score + repro_score
        if total_method_score > 15:
            findings['methodology_quality'] = 'excellent'
        elif total_method_score > 8:
            findings['methodology_quality'] = 'good'
        elif total_method_score > 3:
            findings['methodology_quality'] = 'adequate'
        else:
            findings['methodology_quality'] = 'limited'
        
        return findings
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate various quality and readability metrics"""
        findings = {}
        
        # Basic metrics
        findings['word_count'] = self.metadata.word_count
        findings['complexity_score'] = self.metadata.complexity_score
        
        # Readability metrics (simplified)
        sentences = re.split(r'[.!?]+', self.paper_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            findings['avg_sentence_length'] = avg_sentence_length
            findings['sentence_count'] = len(sentences)
            
            # Readability score (simplified Flesch-like)
            readability = 206.835 - (1.015 * avg_sentence_length)
            findings['readability_score'] = max(0, min(100, readability))
        else:
            findings['avg_sentence_length'] = 0
            findings['sentence_count'] = 0
            findings['readability_score'] = 0
        
        # Technical density
        technical_terms = len(self.metadata.keywords)
        findings['technical_density'] = technical_terms / (self.metadata.word_count / 100) if self.metadata.word_count > 0 else 0
        
        return findings
    
    def _analyze_ml_architecture(self) -> Dict[str, Any]:
        """Analyze machine learning model architecture (domain-specific)"""
        findings = {}
        
        text_lower = self.paper_text.lower()
        
        # ML architecture terms
        ml_terms = [
            'neural network', 'deep learning', 'cnn', 'rnn', 'lstm', 'transformer',
            'attention', 'layer', 'activation', 'optimizer', 'loss function'
        ]
        
        ml_score = sum(text_lower.count(term) for term in ml_terms)
        findings['ml_architecture_mentions'] = ml_score
        
        # Model complexity indicators
        complexity_terms = ['parameter', 'layer', 'hidden', 'dimension', 'feature']
        complexity_score = sum(text_lower.count(term) for term in complexity_terms)
        findings['model_complexity_indicators'] = complexity_score
        
        # Training methodology
        training_terms = ['training', 'epoch', 'batch', 'learning rate', 'gradient']
        training_score = sum(text_lower.count(term) for term in training_terms)
        findings['training_methodology_score'] = training_score
        
        findings['ml_focus_level'] = 'high' if ml_score > 5 else 'medium' if ml_score > 2 else 'low'
        
        return findings
    
    def _review_statistical_methods(self) -> Dict[str, Any]:
        """Review statistical methods used in the paper"""
        findings = {}
        
        text_lower = self.paper_text.lower()
        
        # Statistical method indicators
        stats_terms = [
            'statistical', 'significance', 'p-value', 'confidence interval',
            'hypothesis', 'correlation', 'regression', 'anova', 't-test'
        ]
        
        stats_score = sum(text_lower.count(term) for term in stats_terms)
        findings['statistical_method_indicators'] = stats_score
        
        # Data analysis terms
        data_terms = ['dataset', 'sample', 'population', 'distribution', 'variance']
        data_score = sum(text_lower.count(term) for term in data_terms)
        findings['data_analysis_score'] = data_score
        
        findings['statistical_rigor'] = 'high' if stats_score > 8 else 'medium' if stats_score > 3 else 'low'
        
        return findings
    
    def _generic_analysis(self, task: AnalysisTask) -> Dict[str, Any]:
        """Generic analysis for unspecified tasks"""
        findings = {}
        
        findings['task_type'] = 'generic'
        findings['description'] = task.description
        findings['analysis_performed'] = 'basic content analysis'
        findings['word_count'] = self.metadata.word_count
        findings['sections_analyzed'] = len(self.metadata.sections)
        
        return findings
    
    def _calculate_task_confidence(self, findings: Dict[str, Any]) -> float:
        """Calculate confidence score for a task based on findings quality"""
        # Base confidence on amount of data found
        data_points = len([v for v in findings.values() if v is not None and v != 0])
        
        # Normalize to 0-1 scale
        confidence = min(1.0, data_points / 10.0)
        
        # Boost confidence if we found meaningful results
        if any(isinstance(v, (int, float)) and v > 0 for v in findings.values()):
            confidence = min(1.0, confidence + 0.2)
        
        return confidence
    
    def _generate_task_recommendations(self, task: AnalysisTask, findings: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on task findings"""
        recommendations = []
        
        if "structure" in task.name.lower():
            if findings.get('has_standard_structure', False):
                recommendations.append("Paper follows standard academic structure")
            else:
                recommendations.append("Consider reorganizing content to follow standard academic structure")
        
        elif "keyword" in task.name.lower():
            if findings.get('technical_keywords', 0) < 3:
                recommendations.append("Consider including more domain-specific technical terms")
            if findings.get('keyword_diversity', 0) < 0.7:
                recommendations.append("Increase keyword diversity to improve discoverability")
        
        elif "methodology" in task.name.lower():
            quality = findings.get('methodology_quality', 'unknown')
            if quality in ['limited', 'adequate']:
                recommendations.append("Strengthen methodology section with more rigorous evaluation")
            if findings.get('reproducibility_score', 0) < 3:
                recommendations.append("Improve reproducibility by providing implementation details")
        
        return recommendations
    
    def _generate_overall_assessment(self, task_results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate overall assessment based on all task results"""
        assessment = {}
        
        # Success rate
        successful_tasks = len([r for r in task_results if r.status == 'completed'])
        assessment['success_rate'] = successful_tasks / len(task_results) if task_results else 0
        
        # Average confidence
        confidences = [r.confidence_score for r in task_results if r.confidence_score > 0]
        assessment['average_confidence'] = statistics.mean(confidences) if confidences else 0
        
        # Key strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for result in task_results:
            if result.status == 'completed':
                # Extract strengths and weaknesses from findings
                findings = result.findings
                
                if 'structure_quality' in findings and findings['structure_quality'] == 'good':
                    strengths.append("Well-structured content organization")
                
                if 'contribution_level' in findings and findings['contribution_level'] == 'high':
                    strengths.append("Strong research contributions")
                
                if 'methodology_quality' in findings and findings['methodology_quality'] in ['limited', 'adequate']:
                    weaknesses.append("Methodology could be more rigorous")
                
                if 'reproducibility_score' in findings and findings['reproducibility_score'] < 3:
                    weaknesses.append("Limited reproducibility information")
        
        assessment['strengths'] = list(set(strengths))
        assessment['weaknesses'] = list(set(weaknesses))
        
        # Overall quality assessment
        if assessment['average_confidence'] > 0.8 and assessment['success_rate'] > 0.8:
            assessment['overall_quality'] = 'excellent'
        elif assessment['average_confidence'] > 0.6 and assessment['success_rate'] > 0.7:
            assessment['overall_quality'] = 'good'
        elif assessment['average_confidence'] > 0.4 and assessment['success_rate'] > 0.5:
            assessment['overall_quality'] = 'adequate'
        else:
            assessment['overall_quality'] = 'needs improvement'
        
        return assessment
    
    def _calculate_quality_score(self, task_results: List[AnalysisResult]) -> float:
        """Calculate overall quality score (0-10)"""
        if not task_results:
            return 0.0
        
        # Base score on task success and confidence
        success_rate = len([r for r in task_results if r.status == 'completed']) / len(task_results)
        confidences = [r.confidence_score for r in task_results if r.confidence_score > 0]
        avg_confidence = statistics.mean(confidences) if confidences else 0
        
        # Combine metrics
        base_score = (success_rate * 0.4 + avg_confidence * 0.6) * 10
        
        # Adjust based on paper characteristics
        complexity_bonus = min(2.0, self.metadata.complexity_score / 5.0)
        
        final_score = min(10.0, base_score + complexity_bonus)
        return final_score
    
    def _generate_recommendations(self, task_results: List[AnalysisResult], overall_assessment: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on complete analysis"""
        recommendations = []
        
        # Collect all task recommendations
        for result in task_results:
            recommendations.extend(result.recommendations)
        
        # Add overall recommendations
        if overall_assessment.get('overall_quality') == 'needs improvement':
            recommendations.append("Consider major revisions to improve overall paper quality")
        
        if len(overall_assessment.get('weaknesses', [])) > len(overall_assessment.get('strengths', [])):
            recommendations.append("Focus on addressing identified weaknesses before publication")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
