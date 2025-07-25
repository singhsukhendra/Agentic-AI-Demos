
"""
SENSE Phase: Information gathering and environment perception
Extracts metadata, identifies research domain, and analyzes paper structure
"""

import re
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from .logger import AgenticLogger

@dataclass
class PaperMetadata:
    """Structured representation of paper metadata"""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    sections: List[str]
    word_count: int
    research_domain: str
    paper_type: str
    complexity_score: float

class SenseAgent:
    """
    Sense Agent: Responsible for perceiving and understanding the research paper
    This agent extracts structured information from raw text input
    """
    
    def __init__(self, logger: AgenticLogger):
        self.logger = logger
        self.research_domains = {
            'machine learning': ['neural', 'learning', 'training', 'model', 'algorithm', 'classification', 'regression'],
            'computer vision': ['image', 'visual', 'detection', 'recognition', 'segmentation', 'opencv', 'cnn'],
            'natural language processing': ['text', 'language', 'nlp', 'sentiment', 'tokenization', 'embedding'],
            'robotics': ['robot', 'autonomous', 'control', 'sensor', 'actuator', 'navigation'],
            'cybersecurity': ['security', 'encryption', 'vulnerability', 'attack', 'defense', 'malware'],
            'data science': ['data', 'analysis', 'statistics', 'visualization', 'mining', 'big data'],
            'software engineering': ['software', 'development', 'testing', 'architecture', 'design pattern'],
            'human-computer interaction': ['user', 'interface', 'usability', 'interaction', 'ux', 'ui'],
            'distributed systems': ['distributed', 'parallel', 'cluster', 'cloud', 'scalability'],
            'theoretical computer science': ['algorithm', 'complexity', 'proof', 'theorem', 'mathematical']
        }
        
        self.paper_types = {
            'survey': ['survey', 'review', 'overview', 'comprehensive', 'systematic'],
            'experimental': ['experiment', 'evaluation', 'performance', 'comparison', 'benchmark'],
            'theoretical': ['theorem', 'proof', 'mathematical', 'formal', 'theoretical'],
            'application': ['application', 'implementation', 'system', 'tool', 'framework'],
            'methodology': ['method', 'approach', 'technique', 'algorithm', 'procedure']
        }
    
    def sense(self, paper_text: str) -> PaperMetadata:
        """
        Main sensing function that extracts all relevant information from paper text
        """
        self.logger.phase_start("SENSE", "Analyzing paper structure and extracting metadata")
        
        # Step 1: Extract basic text properties
        self.logger.step("Extracting basic text properties")
        word_count = len(paper_text.split())
        self.logger.result("Word count", word_count)
        
        # Step 2: Extract title
        self.logger.step("Identifying paper title")
        title = self._extract_title(paper_text)
        self.logger.result("Title", title)
        
        # Step 3: Extract authors
        self.logger.step("Extracting author information")
        authors = self._extract_authors(paper_text)
        self.logger.result("Authors found", len(authors))
        
        # Step 4: Extract abstract
        self.logger.step("Locating and extracting abstract")
        abstract = self._extract_abstract(paper_text)
        self.logger.result("Abstract length", f"{len(abstract)} characters")
        
        # Step 5: Extract keywords
        self.logger.step("Identifying key terms and concepts")
        keywords = self._extract_keywords(paper_text)
        self.logger.result("Keywords extracted", len(keywords))
        
        # Step 6: Identify sections
        self.logger.step("Analyzing document structure")
        sections = self._extract_sections(paper_text)
        self.logger.result("Sections identified", len(sections))
        
        # Step 7: Determine research domain
        self.logger.step("Classifying research domain")
        research_domain = self._classify_domain(paper_text)
        self.logger.result("Research domain", research_domain)
        
        # Step 8: Determine paper type
        self.logger.step("Identifying paper type")
        paper_type = self._classify_paper_type(paper_text)
        self.logger.result("Paper type", paper_type)
        
        # Step 9: Calculate complexity score
        self.logger.step("Calculating complexity metrics")
        complexity_score = self._calculate_complexity(paper_text, sections, keywords)
        self.logger.result("Complexity score", f"{complexity_score:.2f}/10")
        
        # Create metadata object
        metadata = PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            sections=sections,
            word_count=word_count,
            research_domain=research_domain,
            paper_type=paper_type,
            complexity_score=complexity_score
        )
        
        self.logger.phase_end("SENSE", f"Extracted metadata for {paper_type} paper in {research_domain}")
        return metadata
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title from text"""
        lines = text.strip().split('\n')
        # Look for the first substantial line that could be a title
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and not line.lower().startswith(('abstract', 'author', 'keyword')):
                return line
        return "Title not found"
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names from text"""
        authors = []
        lines = text.split('\n')
        
        # Look for author patterns
        author_patterns = [
            r'(?:Author[s]?|By):\s*(.+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
        ]
        
        for line in lines[:20]:  # Check first 20 lines
            for pattern in author_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        # Split by comma and clean
                        author_list = [author.strip() for author in match.split(',')]
                        authors.extend(author_list)
        
        # Remove duplicates and filter valid names
        authors = list(set([author for author in authors if len(author.split()) >= 2]))
        return authors[:5]  # Limit to 5 authors
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text"""
        # Look for abstract section
        abstract_match = re.search(r'(?i)abstract[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|1\.|I\.)|$)', 
                                 text, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()
        
        # If no explicit abstract, take first paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 100:
                return para.strip()[:500]  # Limit length
        
        return "Abstract not found"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Technical terms pattern
        technical_terms = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', text_lower)
        
        # Count frequency
        term_freq = {}
        for term in technical_terms:
            if len(term) > 3:  # Only consider terms longer than 3 characters
                term_freq[term] = term_freq.get(term, 0) + 1
        
        # Get most frequent terms
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [term for term, freq in sorted_terms[:15] if freq > 1]
        
        return keywords
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract section headers from text"""
        sections = []
        
        # Common section patterns
        section_patterns = [
            r'^\s*(\d+\.?\s+[A-Z][A-Za-z\s]+)$',  # Numbered sections
            r'^\s*([A-Z][A-Z\s]+)$',  # ALL CAPS sections
            r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$'  # Title case sections
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 5 <= len(line) <= 50:  # Reasonable section header length
                for pattern in section_patterns:
                    if re.match(pattern, line):
                        sections.append(line)
                        break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sections = []
        for section in sections:
            if section not in seen:
                seen.add(section)
                unique_sections.append(section)
        
        return unique_sections[:10]  # Limit to 10 sections
    
    def _classify_domain(self, text: str) -> str:
        """Classify the research domain based on content"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.research_domains.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return "general computer science"
    
    def _classify_paper_type(self, text: str) -> str:
        """Classify the type of research paper"""
        text_lower = text.lower()
        type_scores = {}
        
        for paper_type, keywords in self.paper_types.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            type_scores[paper_type] = score
        
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return "research paper"
    
    def _calculate_complexity(self, text: str, sections: List[str], keywords: List[str]) -> float:
        """Calculate a complexity score for the paper"""
        score = 0.0
        
        # Length factor (0-3 points)
        word_count = len(text.split())
        if word_count > 5000:
            score += 3
        elif word_count > 2000:
            score += 2
        elif word_count > 1000:
            score += 1
        
        # Section structure (0-2 points)
        if len(sections) > 8:
            score += 2
        elif len(sections) > 4:
            score += 1
        
        # Technical vocabulary (0-3 points)
        technical_indicators = ['algorithm', 'methodology', 'framework', 'implementation', 'evaluation']
        tech_score = sum(1 for indicator in technical_indicators if indicator in text.lower())
        score += min(tech_score, 3)
        
        # Mathematical content (0-2 points)
        math_indicators = ['equation', 'formula', 'theorem', 'proof', 'mathematical']
        math_score = sum(1 for indicator in math_indicators if indicator in text.lower())
        score += min(math_score, 2)
        
        return min(score, 10.0)  # Cap at 10
