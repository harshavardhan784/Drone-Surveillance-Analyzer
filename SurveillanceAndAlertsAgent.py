import logging
from utils.utils import detect_objects, retrieve
from tools.GeminiApiTool import GeminiApiTool
import pandas as pd
from pandas import DataFrame

class MyAgentSetup():
    # fallback_response: str = "Sorry, I couldn't understand."
    # summarization_prompt: str = "Summarize briefly."
    # summarization_token_limit: int = 4000
    gpt_api_key_encrypted: str = ""
    tools: dict = {}



class MyAgent():
    def __init__(self, df):
        self.gpt_tool = None
        self.config = None
        self.df = df

    def setup(self, config=None, data=None):
        self.config = config

        # Decrypt API key
        # api_key = config.get("api_key", "")

        # gemini_conf = config.get("gemini_config", {})
        # gemini_conf["api_key"] = api_key
        

        self.gpt_tool = GeminiApiTool(config)
        logging.info("MyAgent setup complete.")


    def _prepare_prompt1(self, df: pd.DataFrame) -> str:
        return f"""
        I have a dataframe containing objects detected by our security drone surveillance system. Each row represents a detected object with the following structure:

        - object_type: Category of detected object (person, vehicle, animal, etc.)
        - object_details: Specific attributes (color, make, model, clothing description)
        - start_frame: Frame number where the object first appears
        - end_frame: Frame number where the object last appears
        - location: Area of the property where the object was detected (e.g., "main gate", "perimeter fence", "garage")
        - timestamp: Time when the object was detected

        Here's the dataframe for reference:
        {self.df}

        Please analyze this surveillance data and provide:

        1. SECURITY EVENT SUMMARY: Identify and summarize key security events in chronological order, including object movements across locations.

        2. ALERT ANALYSIS: Flag suspicious activities based on these security rules:
          - People detected between 22:00-06:00 near entry points
          - Vehicles remaining stationary for over 30 minutes
          - Multiple entries/exits of the same vehicle within the monitoring period
          - Any unidentified objects near secure areas
          - More than 3 people gathered in restricted areas

        3. STATISTICAL OVERVIEW: Provide counts of each object type detected, duration of appearances, and location frequency.

        4. SECURITY ASSESSMENT: Evaluate the overall security situation based on the analyzed data. Identify patterns and potential vulnerabilities.

        ## OUTPUT FORMAT

        {{
            "Logs/Activity": [{{"activity": "Blue Ford F150 spotted at garage", "timestamp": "00:12:00"}}, {{"activity": "", "timestamp": ""}} ...]
            "Alerts": [{{"alert": "alert1", "timestamp": "timestamp1"}}, {{"alert": "alert2", "timestamp": "timestamp2"}}, ...],
        }}

        ## REQUIREMENTS
        1. Output must follow the given format.
        2. Include all relevant logs and alerts.
        3. Include timestamps precisely.
        4. Only return Logs/Activity and Alerts in the output.
        """

    def _prepare_system_prompt1(self):
        return """
            You are an advanced Drone Security Analyst Agent specifically designed to analyze video surveillance data captured by security drones. Your primary function is to process object detection data, identify security events, and generate actionable insights. You have expertise in:

            1. Security threat assessment and classification
            2. Behavioral pattern recognition in surveillance contexts
            3. Vehicle and person identification
            4. Property security protocols and best practices
            5. Data-driven security reporting

            Your responses should be precise, professional, and security-focused. Format your analysis with clear sections, relevant timestamps. When discussing security events, Maintain a neutral, analytical tone appropriate for security professionals.

            Output all timestamps in 24-hour format (HH:MM:SS) and reference frame numbers precisely when relevant. Use bulleted lists for multiple events and formatted tables when presenting statistical summaries. Always include a "Recommendations" section with actionable next steps for security personnel.
            I need accurate timestamps of these logs/activities, so please make sure to include them in the output. Hours minutes and seconds are required in the output.

            The dataset provided for you is in frames, so you can convert the frame numbers to timestamps based on the video frame rate. For example, if the video runs at 30 frames per second, a frame number of 3600 would correspond to 02:00:00 (2 hours).
            Based on whole video duration, you can calculate the timestamps for each log/activity.
            You should give 24-hour format timestamps in the output, like 00:12:00, 01:15:30, etc. You should not randomly follow the timestamps format.
        """


    def _prepare_prompt2(self, query: str, retrieved_logs: str) -> str:
        return f"""
        I have a logs containing objects detected by our security drone surveillance system. Each row represents a detected object with the following structure:

        - log: Description of the event happened in that duration of time in the video.
        - time: The time at which the event has occured in the video
        - score: The score is the percentage of chances that the given log matches the query

        The user query is : {query}

        Here's a logs for reference:
        {retrieved_logs}

        ## OUTPUT FORMAT 
        THE OUTPUT SHOULD BE STRICTLY BASED ON THE QUERY. WHATEVER THE USER ASKS, YOU ARE SUPPOSED TO PROVIDE THAT [CAN BE A PARAGRAPH, LINE, SUMMARIZATION, ETC.]
        YOU ARE NOT SUPPOSED TO PUT YOUR RECOMMENDATIONS OR YOUR ANALYSIS UNTIL AND UNLESS USER ASKS ABOUT IT IN QUERY.
        
        
        ## STRICT GUIDELINES
        - STRICTLY FOLLOW THE USER QUERY AND REPONSE SHOULD BE COMPLETELY BASED ON IT ONLY.
        """

    def _prepare_system_prompt2(self):
        return """
            You are an advanced Drone Security Analyst Agent specifically designed to analyze video surveillance data captured by security drones. Your primary function is to process object detection data, identify security events, and generate actionable insights. You have expertise in:

            1. Security threat assessment and classification
            2. Behavioral pattern recognition in surveillance contexts
            3. Vehicle and person identification
            4. Property security protocols and best practices
            5. Data-driven security reporting
            6. Finding patters like an analyst and answering analysis queries is your task

        """


    def run(self, input_video_path: str, query: str, df: DataFrame, flag = "detections", output_video_path: str = None, output_csv_path: str = None, Retrieved_logs: str = None):
        
        if flag == "detections":
            # df = detect_objects(input_video_path, output_video_path, output_csv_path)

            system_prompt = self._prepare_system_prompt1()
            user_prompt = self._prepare_prompt1(self.df)

            generated_description_with_system, full_response_with_system = self.gpt_tool.run(input_video_path, system_prompt, user_prompt)

            
            # query = "Man jumping a wall"
            results = retrieve(query, generated_description_with_system)

            print("\nTop matching logs:")
            for match in results:
                print(match)
                print(f"- {match['log']} (time: {match['time']}, score: {match['score']}%)")

            Retrieved_logs = ""
            for match in results:
                Retrieved_logs += f"- {match['log']} (time: {match['time']}, score: {match['score']}%)\n"


            return df, generated_description_with_system, Retrieved_logs
        
        elif flag == "logs":
            # df = detect_objects(input_video_path, output_video_path, output_csv_path)

            system_prompt = self._prepare_system_prompt2()
            user_prompt = self._prepare_prompt2(query, Retrieved_logs)

            generated_description_with_system, full_response_with_system = self.gpt_tool.run(input_video_path, system_prompt, user_prompt)

            
            return None, generated_description_with_system, None