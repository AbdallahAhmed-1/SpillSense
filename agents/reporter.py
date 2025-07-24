# agents/reporter.py

import os
import json
import re
import numpy as np
import pandas as pd
from pprint import pprint
from state.state_utils import OilGasRCAState, VisualizationArtifact
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel
from state.session_data import session

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

class Reporter:
    def __init__(self, llm: BaseLanguageModel = None, mode: str = "csv"):
        self.name = "Reporter"
        self.llm = llm
        self.mode = mode
        self.supported_modes = ["csv", "mat", "jpg"]

    def __call__(self, state: OilGasRCAState, mode: str = None) -> OilGasRCAState:
        # Allow mode override at runtime
        if mode is not None:
            self.mode = mode
        return self.run(state)

    def _clean_json(self, text: str) -> str:
        """
        Remove markdown code fences if present, so json.loads succeeds.
        """
        # Strip leading/trailing whitespace
        text = text.strip()
        # Remove triple-backtick fences
        #  - Leading ``` or ```json
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        #  - Trailing ```
        text = re.sub(r"\n```$", "", text)
        return text.strip()

    def _safe_stringify(self, data) -> str:
        """Safely convert data to string representation with NumPy support."""
        try:
            if data is None:
                return "No data available"
            elif isinstance(data, str):
                return data
            elif isinstance(data, (dict, list)):
                # Use custom encoder for NumPy types
                return json.dumps(data, indent=2, cls=NumpyEncoder, ensure_ascii=False)
            else:
                # For other types, try to convert to dict first if possible
                if hasattr(data, 'to_dict'):
                    return json.dumps(data.to_dict(), indent=2, cls=NumpyEncoder, ensure_ascii=False)
                else:
                    return str(data)
        except (TypeError, ValueError) as e:
            # Fallback: convert to string representation
            try:
                return str(data)
            except Exception:
                return f"Error serializing data: {str(e)}"

    def _get_mode_specific_config(self):
        """Get configuration specific to the current mode."""
        configs = {
            "csv": {
                "title": "Oil & Gas RCA Analysis Report",
                "completion_key": "exploration_completed",
                "author": "Automated RCA System",
                "sections": ["Basic Summary", "Missing Values", "Frequency Distribution", "Outliers"],
                "analysis_context": "oil & gas data analyst"
            },
            "mat": {
                "title": "MATLAB Data Analysis Report",
                "completion_key": "exploration_completed",
                "author": "MATLAB Analysis System",
                "sections": ["Data Overview", "Statistical Analysis", "Signal Processing", "Spectral Analysis"],
                "analysis_context": "MATLAB data analyst"
            },
            "jpg": {
                "title": "Image Analysis Report",
                "completion_key": "image_exploration_completed",
                "author": "Image Analysis System",
                "sections": ["Metrics"],
                "analysis_context": "image data analyst"
            }

        }
        return configs.get(self.mode, configs["csv"])

    def _extract_explorator_data(self, state: OilGasRCAState) -> dict:
        """Extract data for explorator mode (original logic)."""
        return {
            "Basic Summary": state.get("basic_summary_result", ""),
            "Missing Values": self._safe_stringify(state.get("missing_values_result", "")),
            "Frequency Distribution": self._safe_stringify(state.get("frequency_distribution_result", "")),
            "Outliers": state.get("outliers_result", "")
        }

    def _extract_mat_data(self, state: OilGasRCAState) -> dict:
        """Extract data for MATLAB mode."""
        return {
            "Data Overview": self._safe_stringify(state.get("mat_data_overview", "")),
            "Statistical Analysis": self._safe_stringify(state.get("mat_statistical_analysis", "")),
            "Signal Processing": self._safe_stringify(state.get("mat_signal_processing", "")),
            "Spectral Analysis": self._safe_stringify(state.get("mat_spectral_analysis", ""))
        }
        
    def _extract_image_data(self, state: OilGasRCAState) -> dict:
        # state['image_summaries'] is a dict of per-image metrics
        summaries = state.get("image_summaries", {})
        return {"Metrics": self._safe_stringify(summaries)}
    
    def _extract_non_visual_data(self, state: OilGasRCAState) -> dict:
        """Extract non-visual data based on current mode."""
        extractors = {
            "csv": self._extract_explorator_data,
            "mat": self._extract_mat_data,
            "jpg": self._extract_image_data
        }
        
        extractor = extractors.get(self.mode, self._extract_explorator_data)
        data = extractor(state)
        
        # Filter out empty sections
        return {k: v for k, v in data.items() if v and v != "No data available"}

    def get_analysis_prompt(self, section_name: str, raw_data: str, config: dict) -> str:
            """Generate analysis prompt based on mode and section with enhanced specificity."""
            
            # Validate inputs
            if not section_name or not raw_data:
                raise ValueError("Section name and raw data cannot be empty")
            
            analysis_context = config.get('analysis_context', 'data analyst')
            
            # Build base prompt with clearer structure
            base_prompt = f"""You are an expert {analysis_context} with deep domain knowledge.

        TASK: Analyze the '{section_name}' section results and provide a comprehensive professional analysis.

        CONTEXT: This analysis is part of a larger {self.mode.upper()} data processing workflow.

        RAW DATA:
        {raw_data}

        ANALYSIS REQUIREMENTS:
        - Identify key patterns, trends, and anomalies
        - Provide quantitative insights where applicable
        - Explain significance and implications
        - Highlight any limitations or uncertainties
        - Use technical terminology appropriate for the domain

        OUTPUT FORMAT (strict JSON, no markdown formatting):
        {{
            "analysis": "<your comprehensive analysis>",
            "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
            "confidence_level": "<high/medium/low>",
            "recommendations": "<actionable recommendations if applicable>"
        }}
        """

            # Enhanced mode-specific guidance
            mode_guidance = {
                "mat": """
        MATLAB-SPECIFIC FOCUS:
        - Matrix operations and computational efficiency
        - Signal processing characteristics and filtering results
        - Numerical stability and convergence properties
        - Algorithm performance metrics
        - Memory usage and computational complexity
        - Visualization of mathematical relationships""",
                
                "csv": """
        HYPERSPECTRAL DATA FOCUS:
        - Spectral signature analysis and band characteristics
        - Spatial-spectral pattern recognition
        - Classification accuracy and confusion matrices
        - Dimensionality reduction effectiveness
        - Noise levels and data quality assessment
        - Feature extraction and selection results""",
                
                "jpg": """
        IMAGE ANALYSIS FOCUS:
        - Visual pattern recognition and spatial distributions
        - Color composition and spectral representation
        - Image quality and resolution considerations
        - Object detection and segmentation results
        - Comparative analysis with reference standards""",
                
                "default": """
        GENERAL DATA ANALYSIS FOCUS:
        - Statistical distributions and central tendencies
        - Data quality and completeness assessment
        - Correlation patterns and relationships
        - Outlier detection and handling
        - Temporal or sequential patterns if applicable"""
            }
            
            # Add mode-specific guidance
            guidance = mode_guidance.get(self.mode, mode_guidance["default"])
            base_prompt += f"\n{guidance}"
            
            # Add section-specific instructions if available
            section_instructions = config.get('section_instructions', {})
            if section_name in section_instructions:
                base_prompt += f"\n\nSECTION-SPECIFIC INSTRUCTIONS:\n{section_instructions[section_name]}"
            
            return base_prompt

    def get_visual_analysis_prompt(self, viz_name: str, config: dict) -> str:
            """Generate enhanced visual analysis prompt with mode-specific guidance."""
            
            # Validate inputs
            if not viz_name:
                raise ValueError("Visualization name cannot be empty")
            
            analysis_context = config.get('analysis_context', 'data visualization expert')
            
            base_prompt = f"""You are an expert {analysis_context} specializing in scientific data visualization.

        TASK: Analyze the visualization titled "{viz_name}" and provide both a concise caption and detailed analysis.

        VISUALIZATION ANALYSIS REQUIREMENTS:
        - Describe what the visualization shows clearly and accurately
        - Identify key visual patterns, trends, and anomalies
        - Explain the significance of visual elements (colors, shapes, distributions)
        - Comment on data quality and representation effectiveness
        - Provide context-appropriate interpretation

        OUTPUT FORMAT (strict JSON, no markdown formatting):
        {{
            "caption": "<concise, descriptive one-sentence caption>",
            "analysis": "<detailed professional analysis>",
            "visual_elements": "<description of key visual components>",
            "insights": ["<insight 1>", "<insight 2>", "<insight 3>"],
            "data_quality_notes": "<observations about data representation quality>"
        }}
        """

            # Enhanced mode-specific visualization guidance
            viz_guidance = {
                "mat": """
        MATLAB VISUALIZATION FOCUS:
        - Signal characteristics and waveform analysis
        - Frequency domain representations (FFT, spectrograms)
        - Matrix visualizations and heatmaps
        - 3D surface plots and contour analysis
        - Filter responses and system characteristics
        - Convergence plots and iterative algorithm results""",
                
                "jpg": """
        HYPERSPECTRAL IMAGE FOCUS:
        - Spectral curve characteristics and band responses
        - False color composite interpretation
        - Classification maps and accuracy visualization
        - Spatial-spectral relationship patterns
        - Endmember analysis and mixing results
        - Atmospheric correction and preprocessing effects""",
                
                "csv": """
        MAINTENANCE & OPERATIONS FOCUS:
        - Equipment performance trends and degradation patterns
        - Operational efficiency metrics and KPIs
        - Predictive maintenance indicators
        - Safety and compliance visualization
        - Cost-benefit analysis representations
        - Resource utilization and optimization charts""",
                
                "default": """
        GENERAL VISUALIZATION FOCUS:
        - Statistical distributions and summary statistics
        - Time series patterns and seasonal variations
        - Correlation matrices and relationship networks
        - Comparative analysis and benchmarking
        - Geographic or spatial distributions if applicable"""
            }
            
            # Add mode-specific guidance
            guidance = viz_guidance.get(self.mode, viz_guidance["default"])
            base_prompt += f"\n{guidance}"
            
            # Add visualization-specific instructions if available
            viz_instructions = config.get('visualization_instructions', {})
            if viz_name in viz_instructions:
                base_prompt += f"\n\nVIZUALIZATION-SPECIFIC INSTRUCTIONS:\n{viz_instructions[viz_name]}"
            
            return base_prompt
    
    def run(self, state: OilGasRCAState) -> OilGasRCAState:
        print(f"[{self.name}] Starting report generation in '{self.mode}' mode...")
        
        # DEBUG: Check what completion flags exist
        completion_flags = {k: v for k, v in state.items() if 'completed' in k}
        print(f"[{self.name}] DEBUG - Completion flags in state: {completion_flags}")
        
        # DEBUG: Check image-related data
        image_keys = {k: type(v) for k, v in state.items() if 'image' in k}
        print(f"[{self.name}] DEBUG - Image-related keys: {image_keys}")
            
        # Validate mode
        if self.mode not in self.supported_modes:
            error_msg = f"Unsupported mode: {self.mode}. Supported modes: {self.supported_modes}"
            state.setdefault("warnings", []).append(error_msg)
            print(f"[{self.name}] {error_msg}")
            return state

        # Get mode-specific configuration
        config = self._get_mode_specific_config()
        
        # Validation checks
        if not state.get(config["completion_key"]):
            warning = f"Reporter: No {self.mode} data available - skipping report generation"
            state.setdefault("warnings", []).append(warning)
            print(f"[{self.name}] {warning}")
            return state

        report_title = state.get("report_title", config["title"])
        print(f"[{self.name}] Report title: {report_title}")

        # Extract non-visual results based on mode
        non_visual = self._extract_non_visual_data(state)
        print(f"[{self.name}] Non-visual sections: {list(non_visual.keys())}")

        # Visual artifacts (same logic for all modes)
        visuals = state.get("visualizations", [])
        print(f"[{self.name}] Found {len(visuals)} visualizations")
        for i, viz in enumerate(visuals):
            print(f"  {i+1}: {viz.title} - {viz.plot_type}")

        if not non_visual and not visuals:
            warning = f"[{self.name}] ⚠️ Very limited data in {self.mode} mode – generating minimal report anyway"
            print(warning)
            state.setdefault("warnings", []).append(warning)

        report = {
            "title": report_title, 
            "sections": [], 
            "mode": self.mode,
            "author": config["author"]
        }

        # --- Process non-visual sections ---
        for name, raw in non_visual.items():
            if not raw:
                continue

            print(f"[{self.name}] Processing non-visual section: {name}")

            prompt = self.get_analysis_prompt(name, raw, config)
            
            try:
                resp = self.llm.invoke(prompt)
                text = getattr(resp, "content", str(resp)).strip()

                # Clean out fences if the model still added them
                clean = self._clean_json(text)

                try:
                    parsed = json.loads(clean)
                    analysis = parsed["analysis"].strip()
                except json.JSONDecodeError as e:
                    print(f"[{self.name}] ⚠️ JSON parse failed for '{name}': {e}")
                    print("Raw LLM output:\n", text[:200], "...")
                    analysis = text

                report["sections"].append({
                    "section_name": name,
                    "result": raw,
                    "analysis": analysis,
                    "section_type": "non_visual"
                })
                print(f"[{self.name}] ✓ Non-visual section '{name}' completed.")
                
            except Exception as e:
                print(f"[{self.name}] ⚠️ Error processing '{name}': {e}")
                report["sections"].append({
                    "section_name": name,
                    "result": raw,
                    "analysis": f"Error generating analysis: {e}",
                    "section_type": "non_visual"
                })

        # --- Process visual sections ---
        for viz in visuals:
            name = viz.title
            uri = viz.base64_image

            print(f"[{self.name}] Processing visual section: {name}")

            if not uri or not uri.startswith("data:image"):
                print(f"[{self.name}] ⚠️ Invalid image data for '{name}'")
                continue

            prompt = self.get_visual_analysis_prompt(name, config)
            
            try:
                resp = self.llm.invoke(prompt)
                text = getattr(resp, "content", str(resp)).strip()

                # Clean out fences
                clean = self._clean_json(text)

                try:
                    parsed = json.loads(clean)
                    caption = parsed["caption"].strip()
                    analysis = parsed["analysis"].strip()
                except json.JSONDecodeError as e:
                    print(f"[{self.name}] ⚠️ JSON parse failed for visualization '{name}': {e}")
                    print("Raw LLM output:\n", text[:200], "...")
                    caption, analysis = f"Visualization: {name}", text

                report["sections"].append({
                    "section_name": name,
                    "image": uri,
                    "caption": caption,
                    "analysis": analysis,
                    "section_type": "visual"
                })
                print(f"[{self.name}] ✓ Visual section '{name}' completed.")
                
            except Exception as e:
                print(f"[{self.name}] ⚠️ Error processing visualization '{name}': {e}")
                report["sections"].append({
                    "section_name": name,
                    "image": uri,
                    "caption": f"Visualization: {name}",
                    "analysis": f"Error generating analysis: {e}",
                    "section_type": "visual"
                })

        # Save final report dict
        state["report_dict"] = report
        print(f"[{self.name}] Report has {len(report['sections'])} sections")

        # Generate PDF from the report dict
        try:
            pdf_path = self._generate_pdf_from_report(report, output_dir=state.get("pdf_report_path", "."))
            state["pdf_report_path"] = pdf_path
            session.last_report_path = pdf_path
            print(f"[{self.name}] ✓ PDF saved at: {pdf_path}")
        except Exception as e:
            error_msg = f"PDF generation failed: {e}"
            print(f"[{self.name}] ⚠️ {error_msg}")
            state.setdefault("warnings", []).append(error_msg)

        state["report_generated"] = True

        state.setdefault("agent_messages", []).append({
            "from_agent": self.name, 
            "message": f"Report generated successfully in '{self.mode}' mode."
        })

        return state

    def _generate_pdf_from_report(self, report: dict, output_dir: str = ".") -> str:
        """
        Render the `report` dict into a PDF, safely handling sections
        that may have dicts or missing images.
        """
        import io, base64, os, re, json
        from datetime import datetime
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        # Prepare filename with mode information
        safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", report.get("title", "report").lower())
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = f"_{self.mode}" if self.mode != "explorator" else ""
        
        filename = f"{safe_title}{mode_suffix}_{ts}.pdf"
        filepath = os.path.join(output_dir, filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Document & styles
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style   = ParagraphStyle("Title",   parent=styles["Heading1"], fontSize=24, spaceAfter=20)
        section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=18, spaceAfter=12)
        caption_style = ParagraphStyle("Caption", parent=styles["Italic"],    fontSize=10,
                                       textColor=colors.gray, spaceAfter=6)
        text_style    = styles["Normal"]

        story = []
        # Title
        story.append(Paragraph(str(report.get("title", "")), title_style))
        story.append(Spacer(1, 12))

        report_author = report.get("author", "Automated Analysis System")
        report_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_mode = report.get("mode", self.mode)
        
        meta_style = styles['Normal']
        story.append(Paragraph(f"<b>Author:</b> {report_author}", meta_style))
        story.append(Paragraph(f"<b>Generated at:</b> {report_generated_at}", meta_style))
        story.append(Paragraph(f"<b>Analysis Mode:</b> {report_mode.title()}", meta_style))
        story.append(Spacer(1, 20))
        
        for sec in report.get("sections", []):
            sec_name = sec.get("section_name", "")
            story.append(Paragraph(str(sec_name), section_style))
            story.append(Spacer(1, 6))

            # 1) Non-visual section?
            if "result" in sec:
                # Safely stringify the result and analysis
                raw_result = sec.get("result", "")
                result_text = json.dumps(raw_result, cls=NumpyEncoder) if isinstance(raw_result, (dict, list)) else str(raw_result)
                if result_text and result_text != "No data available":
                    story.append(Paragraph(f"<b>Data:</b> {result_text}", text_style))
                    story.append(Spacer(1, 6))

                raw_analysis = sec.get("analysis", "")
                analysis_text = json.dumps(raw_analysis, cls=NumpyEncoder) if isinstance(raw_analysis, (dict, list)) else str(raw_analysis)
                if analysis_text:
                    story.append(Paragraph(f"<b>Analysis:</b> {analysis_text}", text_style))
                story.append(Spacer(1, 12))
                continue

            # 2) Visual section
            img_uri = sec.get("image")
            if isinstance(img_uri, str) and img_uri.startswith("data:image"):
                # Decode base64
                b64 = img_uri.split(",", 1)[-1]
                try:
                    img_data = base64.b64decode(b64)
                    buf = io.BytesIO(img_data)
                    img = RLImage(buf, width=6*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 6))
                except Exception as e:
                    story.append(Paragraph(f"(Image load error: {e})", caption_style))
                    story.append(Spacer(1, 6))

                # Caption (always string)
                cap = sec.get("caption", "")
                if cap:
                    story.append(Paragraph(f"<i>{str(cap)}</i>", caption_style))

            # 3) Analysis for visual (or fallback for others)
            ana = sec.get("analysis", "")
            ana_text = json.dumps(ana, cls=NumpyEncoder) if isinstance(ana, (dict, list)) else str(ana)
            if ana_text:
                story.append(Paragraph(f"<b>Analysis:</b> {ana_text}", text_style))

            story.append(Spacer(1, 12))

        # Build the PDF
        try:
            doc.build(story)
            return os.path.abspath(filepath)
        except Exception as e:
            print(f"[{self.name}] PDF generation error: {e}")
            raise e
