"""
Report Generator
================
Generates natural language reports from urinalysis results using LLM.
"""

import os
import json
from typing import Dict, Any, Optional

try:
    # Try new OpenAI client (v1.0+)
    from openai import OpenAI
    OPENAI_V1 = True
except ImportError:
    try:
        # Fall back to old API (v0.x)
        import openai
        OPENAI_V1 = False
    except ImportError:
        print("⚠️  OpenAI package not installed. Run: pip install openai")
        raise

from .rag_pipeline import RAGPipeline


class ReportGenerator:
    """
    Generates natural language medical reports from urinalysis data.
    
    Uses LLM (GPT-4, Claude, Gemini, or Llama) with RAG to:
    1. Interpret urinalysis parameters
    2. Retrieve relevant medical knowledge
    3. Generate comprehensive, evidence-based reports
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        temperature: float = 0.3
    ):
        """
        Initialize report generator.
        
        Args:
            model: LLM model to use (gpt-4, gpt-3.5-turbo, etc.)
            api_key: API key (if None, uses OPENAI_API_KEY env var)
            rag_pipeline: RAG pipeline for knowledge retrieval
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client
        if OPENAI_V1:
            self.client = OpenAI(api_key=self.api_key)
        else:
            openai.api_key = self.api_key
            self.client = None
        
        # Initialize or accept RAG pipeline
        self.rag_pipeline = rag_pipeline
        if self.rag_pipeline and self.rag_pipeline.vectorstore is None:
            print("Building RAG vector store...")
            self.rag_pipeline.build_vector_store()
        
        print(f"✓ Report Generator initialized with {model}")
    
    def _create_system_prompt(self) -> str:
        """
        Create system prompt for medical report generation.
        
        Returns:
            System prompt string
        """
        return """You are UroGPT, an AI medical assistant specializing in urinalysis interpretation.

Your role is to:
1. Analyze urinalysis test results
2. Explain each parameter in clear, professional medical language
3. Identify potential infections or abnormalities (especially UTI)
4. Provide evidence-based interpretations using retrieved medical knowledge
5. Suggest appropriate follow-up actions when necessary

Guidelines:
- Use precise medical terminology while remaining accessible
- Reference normal ranges for each parameter
- Highlight abnormal values and their clinical significance
- Base interpretations on current medical evidence
- Include relevant risk factors and clinical context
- Recommend consulting healthcare professionals for diagnosis

Output Format:
- Clear, structured report with sections for each parameter
- Summary of findings
- UTI probability assessment
- Recommendations"""
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Format urinalysis results for prompt.
        
        Args:
            results: Dictionary with test results
            
        Returns:
            Formatted string
        """
        formatted = "Urinalysis Test Results:\n"
        formatted += "=" * 40 + "\n\n"
        
        # Core parameters
        if "glucose" in results:
            formatted += f"Glucose: {results['glucose']} mg/dL (Normal: 0-15 mg/dL)\n"
        if "pH" in results:
            formatted += f"pH: {results['pH']} (Normal: 4.5-8.0)\n"
        if "nitrite" in results:
            formatted += f"Nitrite: {results['nitrite']} mg/dL (Normal: Negative)\n"
        if "lymphocyte" in results:
            formatted += f"Lymphocytes: {results['lymphocyte']} cells/μL (Normal: <5 cells/μL)\n"
        
        formatted += "\n"
        
        # AI predictions
        if "UTI_probability" in results:
            formatted += f"AI-Predicted UTI Probability: {results['UTI_probability']:.1%}\n"
        if "confidence" in results:
            formatted += f"Confidence Score: {results['confidence']:.1%}\n"
        
        return formatted
    
    def generate_report(
        self,
        urinalysis_results: Dict[str, Any],
        use_rag: bool = True,
        patient_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive medical report from urinalysis results.
        
        Args:
            urinalysis_results: Dictionary with test results from UroAI
            use_rag: Whether to use RAG for knowledge retrieval
            patient_context: Optional patient context/history
            
        Returns:
            Dictionary containing:
            {
                "report": str,              # Full natural language report
                "summary": str,             # Brief summary
                "interpretation": dict,     # Structured interpretation
                "recommendations": list,    # Action items
                "retrieved_context": list   # RAG sources (if used)
            }
        """
        # Format results
        results_text = self._format_results(urinalysis_results)
        
        # Build prompt
        messages = [
            {"role": "system", "content": self._create_system_prompt()}
        ]
        
        # Add patient context if provided
        user_message = results_text
        if patient_context:
            user_message = f"Patient Context:\n{patient_context}\n\n{user_message}"
        
        # Add RAG context if enabled
        retrieved_docs = []
        if use_rag and self.rag_pipeline:
            query = f"urinalysis UTI glucose pH nitrite lymphocyte interpretation"
            context = self.rag_pipeline.get_context(query)
            retrieved_docs = self.rag_pipeline.retrieve(query)
            
            user_message += f"\n\nRelevant Medical Knowledge:\n{context}"
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate report using LLM
        try:
            if OPENAI_V1:
                # New API (v1.0+)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2000
                )
                report = response.choices[0].message.content
            else:
                # Old API (v0.x)
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2000
                )
                report = response.choices[0].message['content']
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            report = self._generate_fallback_report(urinalysis_results)
        
        # Parse and structure response
        result = {
            "report": report,
            "summary": self._extract_summary(report),
            "interpretation": self._extract_interpretation(urinalysis_results),
            "recommendations": self._extract_recommendations(report),
            "retrieved_context": [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
        }
        
        return result
    
    def _generate_fallback_report(self, results: Dict[str, Any]) -> str:
        """
        Generate basic report if LLM call fails.
        
        Args:
            results: Urinalysis results
            
        Returns:
            Basic report string
        """
        report = "Urinalysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Glucose
        glucose = results.get("glucose", 0)
        report += f"Glucose: {glucose} mg/dL\n"
        if glucose > 15:
            report += "  ⚠️ ABNORMAL - Elevated glucose may indicate diabetes or kidney issues\n"
        else:
            report += "  ✓ Normal range\n"
        report += "\n"
        
        # pH
        ph = results.get("pH", 7.0)
        report += f"pH: {ph}\n"
        if ph < 4.5 or ph > 8.0:
            report += "  ⚠️ ABNORMAL - pH outside normal range\n"
        else:
            report += "  ✓ Normal range\n"
        report += "\n"
        
        # Nitrite
        nitrite = results.get("nitrite", 0)
        report += f"Nitrite: {nitrite} mg/dL\n"
        if nitrite > 0:
            report += "  ⚠️ POSITIVE - Suggests bacterial infection (UTI)\n"
        else:
            report += "  ✓ Negative\n"
        report += "\n"
        
        # Lymphocytes
        lymphocyte = results.get("lymphocyte", 0)
        report += f"Lymphocytes: {lymphocyte} cells/μL\n"
        if lymphocyte > 5:
            report += "  ⚠️ ELEVATED - May indicate infection or inflammation\n"
        else:
            report += "  ✓ Normal range\n"
        report += "\n"
        
        # UTI probability
        uti_prob = results.get("UTI_probability", 0)
        report += f"\nUTI Probability: {uti_prob:.1%}\n"
        if uti_prob > 0.7:
            report += "⚠️  HIGH RISK - Strong indicators of urinary tract infection\n"
        elif uti_prob > 0.4:
            report += "⚠️  MODERATE RISK - Some indicators present\n"
        else:
            report += "✓ LOW RISK\n"
        
        report += "\nRecommendations:\n"
        report += "- Consult with healthcare provider for proper diagnosis\n"
        if uti_prob > 0.5:
            report += "- Consider antibiotic treatment if UTI confirmed\n"
            report += "- Increase fluid intake\n"
        
        return report
    
    def _extract_summary(self, report: str) -> str:
        """
        Extract brief summary from full report.
        
        Args:
            report: Full report text
            
        Returns:
            Summary string
        """
        lines = report.split("\n")
        summary_lines = []
        
        for line in lines[:10]:  # Take first few lines
            if line.strip():
                summary_lines.append(line.strip())
        
        return " ".join(summary_lines[:3])
    
    def _extract_interpretation(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Create structured interpretation of each parameter.
        
        Args:
            results: Test results
            
        Returns:
            Dictionary with interpretations
        """
        interpretation = {}
        
        # Glucose
        glucose = results.get("glucose", 0)
        interpretation["glucose"] = "Normal" if glucose <= 15 else "Elevated"
        
        # pH
        ph = results.get("pH", 7.0)
        interpretation["pH"] = "Normal" if 4.5 <= ph <= 8.0 else "Abnormal"
        
        # Nitrite
        nitrite = results.get("nitrite", 0)
        interpretation["nitrite"] = "Negative" if nitrite == 0 else "Positive"
        
        # Lymphocytes
        lymphocyte = results.get("lymphocyte", 0)
        interpretation["lymphocyte"] = "Normal" if lymphocyte < 5 else "Elevated"
        
        # UTI
        uti_prob = results.get("UTI_probability", 0)
        if uti_prob > 0.7:
            interpretation["UTI_risk"] = "High"
        elif uti_prob > 0.4:
            interpretation["UTI_risk"] = "Moderate"
        else:
            interpretation["UTI_risk"] = "Low"
        
        return interpretation
    
    def _extract_recommendations(self, report: str) -> list:
        """
        Extract recommendations from report.
        
        Args:
            report: Full report text
            
        Returns:
            List of recommendations
        """
        recommendations = [
            "Consult healthcare provider for proper diagnosis",
            "Monitor symptoms and follow up if condition worsens"
        ]
        
        return recommendations

