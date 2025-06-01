import google.generativeai as genai
from utils.utils import wait_for_file_activation
import os
import pandas as pd
from google.generativeai import upload_file
from google.ai.generativelanguage_v1beta.types import Content, Part
from google.protobuf import struct_pb2
import pandas as pd
import mimetypes
import time
import json
from typing import Dict, Any, Tuple, Optional
from pandas import DataFrame
from SurveillanceAndAlertsAgent import MyAgent as SurveillanceAndAlertsAgent
from TelemetryAndQueryAgent import MyAgent as TelemetryAndQueryAgent


class OrchestratorTool:
    """Main orchestrator that routes queries to appropriate specialized agents"""
    
    def __init__(self, config: dict, df: DataFrame):
        self.config = config
        self.model_name = config.get("routing_model", "gemini-1.5-flash")  # Lighter model for routing
        self.api_key = config.get("api_key")
        self.df = df 

        if not self.api_key:
            raise ValueError("API key missing for OrchestratorTool")
        
        genai.configure(api_key=self.api_key)
        self.routing_model = genai.GenerativeModel(self.model_name)
        
        # Initialize specialized agents
        self.surveillance_agent = SurveillanceAndAlertsAgent(df)
        self.telemetry_agent = TelemetryAndQueryAgent(df)

        self.surveillance_agent.setup(config)
        self.telemetry_agent.setup(config)
        
        # Routing prompt for agent selection
        self.routing_prompt = """
        You are a routing agent that determines which specialized agent should handle a video analysis query.
        
        Available agents:
        1. SURVEILLANCE_AGENT - For security, safety, surveillance, threats, anomalies, alerts, incidents, suspicious activities
        2. TELEMETRY_AGENT - For data analysis, measurements, statistics, object detection, motion analysis, general queries
        
        Based on the user query, respond with ONLY the agent name (SURVEILLANCE_AGENT or TELEMETRY_AGENT).
        
        Examples:
        - "Detect any suspicious activities" → SURVEILLANCE_AGENT
        - "Is there any security threat?" → SURVEILLANCE_AGENT  
        - "Generate alerts for unusual behavior" → SURVEILLANCE_AGENT
        - "Count the number of people" → TELEMETRY_AGENT
        - "Analyze the motion patterns" → TELEMETRY_AGENT
        - "What objects are visible?" → TELEMETRY_AGENT
        - "Describe what's happening" → TELEMETRY_AGENT
        
        User Query: {query}
        
        Agent:"""


        # Configuration
        # self.VIDEO_PATH = ".\videos\combined.mp4"
        # self.QUERY_FOR_DETECTION = "railway track"
        # self.Ask_any_question = "how many vechiles passed through railway gate."
        self.OUTPUT_VIDEO = "output.mp4"
        self.OUTPUT_CSV = "detections.csv"
    
    def _route_query(self, user_prompt: str) -> str:
        """Determine which agent should handle the query"""
        try:
            prompt =  """ {{
                Take atmost care for deciding which agent should handle the query.
                "SURVEILLANCE_AGENT": "Handles security monitoring, summarizing video, threat detection, anomaly detection, alert generation, incident analysis, and safety assessments",
                "TELEMETRY_AGENT": "Handles data analysis, performance metrics, statistical analysis. If the query is related to timestamps or asking at what time a certain event happened, then it will be handled by Telemetry Agent. Counting no of people or obect in the video, then it will be handled by Telemetry Agent.",

                "At what timestamp this event has occured, all particular event related queries are given to Telemetry Agent.": "TELEMETRY_AGENT"
            }}
            """
            routing_query = self.routing_prompt.format(query=prompt + user_prompt)
            response = self.routing_model.generate_content(
                routing_query,
                generation_config={"temperature": 0.1, "max_output_tokens": 50}
            )
            
            if response.candidates:
                agent_choice = response.candidates[0].content.parts[0].text.strip().upper()
                if "SURVEILLANCE_AGENT" in agent_choice:
                    return "SURVEILLANCE_AGENT"
                elif "TELEMETRY_AGENT" in agent_choice:
                    return "TELEMETRY_AGENT"
                else:
                    # Default to telemetry agent for general queries
                    return "TELEMETRY_AGENT"
            else:
                return "TELEMETRY_AGENT"
                
        except Exception as e:
            print(f"Error in routing decision: {e}")
            # Default to telemetry agent on error
            return "TELEMETRY_AGENT"
    
    def run(self, video_path: str, system_prompt: str, user_prompt: str) -> Tuple[str, Any, str]:
        """
        Main orchestration method that routes queries to appropriate agents
        
        Returns:
            Tuple[str, Any, str]: (response_text, api_response, selected_agent)
        """
        print(f"[Orchestrator] Processing query: {user_prompt[:100]}...")
        
        # Route the query to appropriate agent
        selected_agent = self._route_query(user_prompt)
        print(f"[Orchestrator] Routing to: {selected_agent}")
        
        
        # Execute with selected agent
        if selected_agent == "SURVEILLANCE_AGENT":
            # For surveillance, we might want to override or enhance the system prompt
            enhanced_prompt = f"{system_prompt}\n\nFOCUS ON SECURITY AND SURVEILLANCE ASPECTS." if system_prompt else ""
            df, generated_description_with_system, Retrieved_logs = self.surveillance_agent.run(video_path, user_prompt, self.df, "detections", self.OUTPUT_VIDEO, self.OUTPUT_CSV)
        else:  # TELEMETRY_AGENT
            # For telemetry, we might want to override or enhance the system prompt
            enhanced_prompt = f"{system_prompt}\n\nFOCUS ON DATA ANALYSIS AND DETAILED INSIGHTS and gives details on telemetry." if system_prompt else ""
            df, generated_description_with_system, Retrieved_logs = self.telemetry_agent.run(video_path, user_prompt, self.df, "detections", self.OUTPUT_VIDEO, self.OUTPUT_CSV)
            df, generated_description_with_system, Retrieved_logs = self.telemetry_agent.run(video_path, user_prompt, self.df, "logs", self.OUTPUT_VIDEO, self.OUTPUT_CSV)
        
        print(f"[Orchestrator] Task completed by {selected_agent}")
        return (generated_description_with_system, Retrieved_logs, selected_agent)
    
    def get_agent_capabilities(self) -> Dict[str, str]:
        """Return information about agent capabilities"""
        return {
            "SURVEILLANCE_AGENT": "Handles security monitoring, summarizing video, threat detection, anomaly detection, alert generation, incident analysis, and safety assessments",
            "TELEMETRY_AGENT": "Handles data analysis, performance metrics, statistical analysis. If the query is related to timestamps or asking at what time a certain event happened, then it will be handled by Telemetry Agent.",
        }

