4096
"""
AI-Powered Pipeline for Research Paper Analysis
Implements Sense-Plan-Act cycle using LLMs for actual reasoning
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .llm_client import LLMClient, create_llm_client, LLMClientError
from .logger import AgenticLogger

@dataclass
class AIAnalysisResult:
    """Result from AI analysis with LLM reasoning"""
    metadata_summary: Dict[str, Any]
    execution_summary: Dict[str, Any]
    overall_assessment: Dict[str, Any]
    quality_score: float
    recommendations: List[str]
    task_results: List[Dict[str, Any]]
    llm_reasoning: Dict[str, str]  # Store LLM reasoning for each phase
    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0

class AISenseAgent:
    """AI-powered sensing agent using LLM for paper analysis"""

    def __init__(self, llm_client: LLMClient, logger: AgenticLogger):
        self.llm_client = llm_client
        self.logger = logger

    def sense(self, paper_text: str) -> Dict[str, Any]:
        """Use LLM to analyze and extract insights from paper"""

        self.logger.phase_start("SENSE", "AI-powered paper analysis and insight extraction")

        system_prompt = """You are an expert research paper analyzer. Your task is to carefully analyze the given research paper and extract comprehensive metadata and insights.

Please analyze the paper and provide a detailed JSON response with the following structure:
{
    "title": "extracted title",
    "authors": ["list of authors"],
    "abstract": "paper abstract if available",
    "paper_type": "research/survey/position/technical report/etc",
    "research_domain": "specific field like 'machine learning', 'computer vision', etc",
    "keywords": ["extracted keywords"],
    "sections": ["list of main sections"],
    "word_count": estimated_word_count,
    "complexity_score": score_from_1_to_10,
    "methodology": "brief description of methodology used",
    "key_contributions": ["list of main contributions"],
    "datasets_used": ["datasets mentioned"],
    "evaluation_metrics": ["metrics used for evaluation"],
    "limitations": ["identified limitations"],
    "future_work": ["suggested future directions"],
    "citations_quality": "assessment of citation quality",
    "writing_quality": "assessment of writing clarity",
    "novelty_assessment": "assessment of novelty and originality",
    "technical_depth": "assessment of technical depth",
    "reproducibility": "assessment of reproducibility"
}

Be thorough and analytical. If information is not available, use "N/A" or appropriate defaults."""

        user_prompt = f"""Please analyze this research paper and extract comprehensive metadata:

PAPER TEXT:
{paper_text[:4096]}  # Limit to avoid token limits

Provide your analysis as a valid JSON object with detailed insights."""

        try:
            start_time = time.time()
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=4096
            )

            # Parse LLM response
            try:
                metadata = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response if wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")

            # Add LLM response metadata
            metadata['llm_analysis_time'] = time.time() - start_time
            metadata['llm_tokens_used'] = response.tokens_used or 0
            metadata['llm_provider'] = response.provider
            metadata['llm_model'] = response.model

            self.logger.step_complete("Paper metadata extraction",
                                    f"Extracted {len(metadata)} metadata fields using {response.provider}")

            # Log key insights
            self.logger.info(f"Paper Type: {metadata.get('paper_type', 'Unknown')}")
            self.logger.info(f"Research Domain: {metadata.get('research_domain', 'Unknown')}")
            self.logger.info(f"Complexity Score: {metadata.get('complexity_score', 'N/A')}/10")
            self.logger.info(f"Key Contributions: {len(metadata.get('key_contributions', []))}")

            self.logger.phase_end("SENSE", time.time() - start_time)

            return metadata

        except Exception as e:
            self.logger.error(f"AI sensing failed: {str(e)}")
            # Return minimal metadata as fallback
            return {
                'title': 'Unknown',
                'paper_type': 'unknown',
                'research_domain': 'unknown',
                'word_count': len(paper_text.split()),
                'complexity_score': 5.0,
                'error': str(e)
            }

class AIPlanAgent:
    """AI-powered planning agent using LLM for analysis strategy"""

    def __init__(self, llm_client: LLMClient, logger: AgenticLogger):
        self.llm_client = llm_client
        self.logger = logger

    def plan(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to create analysis strategy based on paper characteristics"""

        self.logger.phase_start("PLAN", "AI-powered analysis strategy creation")

        system_prompt = """You are an expert research analysis strategist. Based on the paper metadata provided, create a comprehensive analysis plan that will guide the detailed evaluation of this research paper.

Your plan should be tailored to the specific type of paper, research domain, and characteristics identified. Provide a detailed JSON response with this structure:

{
    "analysis_strategy": "overall strategy description",
    "priority_tasks": [
        {
            "task_name": "specific task name",
            "description": "what this task will analyze",
            "importance": "high/medium/low",
            "estimated_time": "time estimate in seconds",
            "specific_focus": "what to focus on for this paper type",
            "evaluation_criteria": ["list of criteria to evaluate"],
            "expected_insights": ["what insights this task should provide"]
        }
    ],
    "domain_specific_considerations": ["considerations specific to this research domain"],
    "quality_assessment_framework": {
        "technical_rigor": "how to assess technical quality",
        "novelty_evaluation": "how to evaluate novelty",
        "impact_assessment": "how to assess potential impact",
        "reproducibility_check": "how to check reproducibility"
    },
    "risk_factors": ["potential issues to watch for"],
    "success_metrics": ["how to measure analysis success"],
    "adaptation_notes": "how this plan is adapted for this specific paper"
}

Be strategic and thorough. Consider the paper type, domain, and complexity when creating the plan."""

        user_prompt = f"""Based on this paper metadata, create a comprehensive analysis plan:

PAPER METADATA:
{json.dumps(metadata, indent=2)}

Create a strategic analysis plan tailored to this specific paper's characteristics."""

        try:
            start_time = time.time()
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,  # Slightly higher for creative planning
                max_tokens=4096
            )

            # Parse LLM response
            try:
                plan = json.loads(response.content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")

            # Add planning metadata
            plan['planning_time'] = time.time() - start_time
            plan['llm_tokens_used'] = response.tokens_used or 0
            plan['llm_provider'] = response.provider
            plan['llm_model'] = response.model

            # Log planning insights
            num_tasks = len(plan.get('priority_tasks', []))
            high_priority_tasks = len([t for t in plan.get('priority_tasks', [])
                                     if t.get('importance') == 'high'])

            self.logger.step_complete("Analysis strategy creation",
                                    f"Created plan with {num_tasks} tasks ({high_priority_tasks} high priority)")

            self.logger.info(f"Strategy: {plan.get('analysis_strategy', 'N/A')}")
            self.logger.info(f"Domain considerations: {len(plan.get('domain_specific_considerations', []))}")

            self.logger.phase_end("PLAN", time.time() - start_time)

            return plan

        except Exception as e:
            self.logger.error(f"AI planning failed: {str(e)}")
            # Return basic plan as fallback
            return {
                'analysis_strategy': 'Basic analysis due to planning error',
                'priority_tasks': [
                    {
                        'task_name': 'Basic Quality Assessment',
                        'description': 'Perform basic paper quality assessment',
                        'importance': 'high',
                        'estimated_time': '30'
                    }
                ],
                'error': str(e)
            }

class AIActAgent:
    """AI-powered action agent using LLM for comprehensive analysis execution"""

    def __init__(self, llm_client: LLMClient, logger: AgenticLogger):
        self.llm_client = llm_client
        self.logger = logger

    def act(self, paper_text: str, metadata: Dict[str, Any], plan: Dict[str, Any]) -> AIAnalysisResult:
        """Execute comprehensive analysis using LLM reasoning"""

        self.logger.phase_start("ACT", "AI-powered comprehensive analysis execution")

        total_tokens = 0
        total_cost = 0.0
        task_results = []
        llm_reasoning = {}

        # Execute each planned task
        priority_tasks = plan.get('priority_tasks', [])

        for i, task in enumerate(priority_tasks, 1):
            task_name = task.get('task_name', f'Task {i}')
            self.logger.info(f"Executing task {i}/{len(priority_tasks)}: {task_name}")

            try:
                task_result = self._execute_analysis_task(paper_text, metadata, task, plan)
                task_results.append(task_result)

                if 'llm_tokens_used' in task_result:
                    total_tokens += task_result['llm_tokens_used']
                if 'llm_cost_estimate' in task_result:
                    total_cost += task_result['llm_cost_estimate']

                # Store LLM reasoning
                if 'llm_reasoning' in task_result:
                    llm_reasoning[task_name] = task_result['llm_reasoning']

            except Exception as e:
                self.logger.error(f"Task {task_name} failed: {str(e)}")
                task_results.append({
                    'task_name': task_name,
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': 0,
                    'confidence_score': 0.0
                })

        # Generate overall assessment using LLM
        overall_assessment = self._generate_overall_assessment(paper_text, metadata, task_results, plan)
        if 'llm_tokens_used' in overall_assessment:
            total_tokens += overall_assessment['llm_tokens_used']
        if 'llm_reasoning' in overall_assessment:
            llm_reasoning['overall_assessment'] = overall_assessment['llm_reasoning']

        # Generate recommendations using LLM
        recommendations = self._generate_recommendations(paper_text, metadata, task_results, overall_assessment)
        if 'llm_tokens_used' in recommendations:
            total_tokens += recommendations['llm_tokens_used']
        if 'llm_reasoning' in recommendations:
            llm_reasoning['recommendations'] = recommendations['llm_reasoning']

        # Calculate quality score
        quality_score = self._calculate_quality_score(task_results, overall_assessment)

        # Create execution summary
        execution_summary = {
            'total_tasks': len(priority_tasks),
            'completed_tasks': len([r for r in task_results if r.get('status') == 'completed']),
            'failed_tasks': len([r for r in task_results if r.get('status') == 'failed']),
            'average_confidence': sum(r.get('confidence_score', 0) for r in task_results) / len(task_results) if task_results else 0,
            'total_execution_time': sum(r.get('execution_time', 0) for r in task_results),
            'success_rate': len([r for r in task_results if r.get('status') == 'completed']) / len(task_results) if task_results else 0
        }

        self.logger.phase_end("ACT", execution_summary['total_execution_time'])

        return AIAnalysisResult(
            metadata_summary=metadata,
            execution_summary=execution_summary,
            overall_assessment=overall_assessment,
            quality_score=quality_score,
            recommendations=recommendations.get('recommendations', []),
            task_results=task_results,
            llm_reasoning=llm_reasoning,
            total_tokens_used=total_tokens,
            total_cost_estimate=total_cost
        )

    def _execute_analysis_task(self, paper_text: str, metadata: Dict[str, Any],
                              task: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific analysis task using LLM"""

        task_name = task.get('task_name', 'Unknown Task')
        start_time = time.time()

        # Create task-specific system prompt
        system_prompt = f"""You are an expert research paper analyst executing a specific analysis task.

TASK: {task_name}
DESCRIPTION: {task.get('description', 'No description provided')}
FOCUS: {task.get('specific_focus', 'General analysis')}
EVALUATION CRITERIA: {', '.join(task.get('evaluation_criteria', []))}

Provide a detailed analysis in JSON format:
{{
    "findings": {{
        "key_finding_1": "detailed finding",
        "key_finding_2": "detailed finding",
        ...
    }},
    "assessment": "overall assessment for this task",
    "confidence_score": score_from_0_to_1,
    "evidence": ["supporting evidence from the paper"],
    "concerns": ["any concerns or limitations identified"],
    "task_specific_insights": ["insights specific to this analysis task"],
    "reasoning": "detailed explanation of your analysis process and conclusions"
}}

Be thorough, analytical, and provide specific evidence from the paper."""

        # Create task-specific user prompt
        user_prompt = f"""Analyze this research paper for the specific task: {task_name}

PAPER METADATA:
{json.dumps({k: v for k, v in metadata.items() if k not in ['llm_analysis_time', 'llm_tokens_used']}, indent=2)}

PAPER TEXT (first 6000 chars):
{paper_text[:6000]}

ANALYSIS PLAN CONTEXT:
Strategy: {plan.get('analysis_strategy', 'N/A')}
Domain Considerations: {plan.get('domain_specific_considerations', [])}

Execute the analysis task with focus on: {task.get('specific_focus', 'general analysis')}"""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=4096
            )

            # Parse response
            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: create structured result from text
                    result = {
                        'findings': {'analysis': response.content},
                        'assessment': 'Analysis completed with text response',
                        'confidence_score': 0.7,
                        'reasoning': response.content
                    }

            execution_time = time.time() - start_time

            # Add task metadata
            task_result = {
                'task_name': task_name,
                'status': 'completed',
                'execution_time': execution_time,
                'findings': result.get('findings', {}),
                'assessment': result.get('assessment', 'No assessment provided'),
                'confidence_score': result.get('confidence_score', 0.5),
                'evidence': result.get('evidence', []),
                'concerns': result.get('concerns', []),
                'task_specific_insights': result.get('task_specific_insights', []),
                'llm_reasoning': result.get('reasoning', ''),
                'llm_tokens_used': response.tokens_used or 0,
                'llm_provider': response.provider,
                'llm_model': response.model
            }

            self.logger.step_complete(task_name,
                                    f"Confidence: {task_result['confidence_score']:.2f}, "
                                    f"Findings: {len(task_result['findings'])}")

            return task_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task execution failed: {str(e)}")

            return {
                'task_name': task_name,
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(e),
                'confidence_score': 0.0,
                'findings': {}
            }

    def _generate_overall_assessment(self, paper_text: str, metadata: Dict[str, Any],
                                   task_results: List[Dict[str, Any]], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment using LLM synthesis"""

        system_prompt = """You are an expert research paper evaluator. Based on the detailed task analyses provided, synthesize an overall assessment of this research paper.

Provide a comprehensive JSON assessment:
{
    "overall_quality": "excellent/good/fair/poor",
    "strengths": ["list of key strengths"],
    "weaknesses": ["list of key weaknesses"],
    "technical_rigor": "assessment of technical quality",
    "novelty_score": score_from_1_to_10,
    "impact_potential": "assessment of potential impact",
    "reproducibility_assessment": "assessment of reproducibility",
    "writing_quality": "assessment of clarity and presentation",
    "contribution_significance": "assessment of contribution significance",
    "methodology_soundness": "assessment of methodology",
    "evaluation_adequacy": "assessment of evaluation approach",
    "related_work_coverage": "assessment of related work",
    "success_rate": calculated_success_rate,
    "average_confidence": calculated_average_confidence,
    "synthesis_reasoning": "detailed explanation of your overall assessment"
}

Be objective, balanced, and provide specific justifications."""

        # Prepare task results summary
        task_summary = []
        for result in task_results:
            if result.get('status') == 'completed':
                task_summary.append({
                    'task': result['task_name'],
                    'assessment': result.get('assessment', ''),
                    'confidence': result.get('confidence_score', 0),
                    'key_findings': list(result.get('findings', {}).keys())
                })

        user_prompt = f"""Synthesize an overall assessment based on these detailed analyses:

PAPER METADATA:
{json.dumps({k: v for k, v in metadata.items() if not k.startswith('llm_')}, indent=2)}

TASK ANALYSIS RESULTS:
{json.dumps(task_summary, indent=2)}

DETAILED FINDINGS:
{json.dumps([{r['task_name']: r.get('findings', {})} for r in task_results if r.get('status') == 'completed'], indent=2)}

Provide a comprehensive overall assessment that synthesizes all the task analyses."""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=4096
            )

            try:
                assessment = json.loads(response.content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    assessment = json.loads(json_match.group())
                else:
                    assessment = {
                        'overall_quality': 'fair',
                        'strengths': ['Analysis completed'],
                        'weaknesses': ['Assessment parsing failed'],
                        'synthesis_reasoning': response.content
                    }

            # Add LLM metadata
            assessment['llm_tokens_used'] = response.tokens_used or 0
            assessment['llm_reasoning'] = assessment.get('synthesis_reasoning', '')

            return assessment

        except Exception as e:
            return {
                'overall_quality': 'unknown',
                'strengths': [],
                'weaknesses': [f'Assessment generation failed: {str(e)}'],
                'error': str(e)
            }

    def _generate_recommendations(self, paper_text: str, metadata: Dict[str, Any],
                                task_results: List[Dict[str, Any]], overall_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations using LLM"""

        system_prompt = """You are an expert research mentor providing actionable recommendations for improving a research paper. Based on the comprehensive analysis provided, generate specific, actionable recommendations.

Provide recommendations in JSON format:
{
    "recommendations": [
        "specific actionable recommendation 1",
        "specific actionable recommendation 2",
        ...
    ],
    "priority_recommendations": ["top 3 most important recommendations"],
    "improvement_areas": {
        "technical": ["technical improvements needed"],
        "presentation": ["presentation improvements needed"],
        "methodology": ["methodology improvements needed"],
        "evaluation": ["evaluation improvements needed"]
    },
    "future_work_suggestions": ["suggestions for future research directions"],
    "publication_readiness": "assessment of publication readiness",
    "recommendation_reasoning": "detailed explanation of recommendation rationale"
}

Be constructive, specific, and actionable. Focus on improvements that would have the most impact."""

        user_prompt = f"""Based on this comprehensive analysis, provide actionable recommendations:

OVERALL ASSESSMENT:
{json.dumps({k: v for k, v in overall_assessment.items() if not k.startswith('llm_')}, indent=2)}

KEY CONCERNS FROM ANALYSIS:
{json.dumps([r.get('concerns', []) for r in task_results if r.get('concerns')], indent=2)}

PAPER CHARACTERISTICS:
Type: {metadata.get('paper_type', 'Unknown')}
Domain: {metadata.get('research_domain', 'Unknown')}
Complexity: {metadata.get('complexity_score', 'Unknown')}/10

Generate specific, actionable recommendations for improving this research paper."""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=4096
            )

            try:
                recommendations = json.loads(response.content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    recommendations = json.loads(json_match.group())
                else:
                    # Extract recommendations from text
                    rec_lines = [line.strip() for line in response.content.split('\n')
                               if line.strip() and ('recommend' in line.lower() or line.strip().startswith('-'))]
                    recommendations = {
                        'recommendations': rec_lines,
                        'recommendation_reasoning': response.content
                    }

            # Add LLM metadata
            recommendations['llm_tokens_used'] = response.tokens_used or 0
            recommendations['llm_reasoning'] = recommendations.get('recommendation_reasoning', '')

            return recommendations

        except Exception as e:
            return {
                'recommendations': [f'Recommendation generation failed: {str(e)}'],
                'error': str(e)
            }

    def _calculate_quality_score(self, task_results: List[Dict[str, Any]],
                               overall_assessment: Dict[str, Any]) -> float:
        """Calculate overall quality score based on analysis results"""

        if not task_results:
            return 5.0

        # Base score from task confidence
        confidence_scores = [r.get('confidence_score', 0.5) for r in task_results if r.get('status') == 'completed']
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        # Adjust based on overall assessment
        quality_mapping = {
            'excellent': 9.0,
            'good': 7.5,
            'fair': 5.5,
            'poor': 3.0
        }

        overall_quality = overall_assessment.get('overall_quality', 'fair')
        base_score = quality_mapping.get(overall_quality, 5.5)

        # Combine confidence and quality assessment
        final_score = (base_score * 0.7) + (avg_confidence * 10 * 0.3)

        return min(10.0, max(1.0, final_score))

def create_ai_pipeline(config_path: Optional[str] = None) -> tuple:
    """Factory function to create AI pipeline components"""

    # Create LLM client
    llm_client = create_llm_client(config_path)

    # Create logger
    logger = AgenticLogger()

    # Create AI agents
    sense_agent = AISenseAgent(llm_client, logger)
    plan_agent = AIPlanAgent(llm_client, logger)
    act_agent = AIActAgent(llm_client, logger)

    return sense_agent, plan_agent, act_agent, logger
