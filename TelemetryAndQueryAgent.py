import logging
from utils.utils import detect_objects, retrieve
from tools.GeminiApiTool import GeminiApiTool
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2

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
        self.df = df  # Store the dataframe for later use

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

        """


    def _prepare_prompt2(self, query: str, retrieved_logs: str) -> str:
        print("Preparing prompt for retrieved logs:", retrieved_logs)
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
        - The log's contain the timestamp in the format of "MM:SS:MS" where MM is minutes, SS is seconds, and MS is milliseconds. If the query is related to a specific time, you should return the logs that are relevant to that time, but The output should follow both the timestamps given in the logs and according to the video time.
        - If the query is related to a specific time range, you should return the answer and the time w.r.to logs timstamps and video timestamp both that are relevant to that time range.
        - If the query is related to a specific time range, then also return the logs that are relevant to that time range. 
        - #Must => You should also return the logs that are relevant to the query and the time range given in the query.
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
    

    def get_video_length(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return duration


    def get_relevant_segments(self, log_times, video_len, padding=10):
        intervals = []
        for t in log_times:
            if t is None:
                continue

            # Split the string into components
            parts = t.strip().split(":")
            if len(parts) != 3:
                continue

            minutes, seconds, millis_as_seconds = map(int, parts)
            
            # Treat "milliseconds" field as actual SECONDS
            # (Because your input is effectively mm:ss:00, not mm:ss:ms)
            total_seconds = minutes * 60 + seconds + millis_as_seconds

            start = max(0, total_seconds - padding)
            end = min(video_len, total_seconds + padding)
            intervals.append([start, end])

        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])

        # Merge overlapping/adjacent intervals
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0] - 1e-6:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged



    def extract_and_combine_segments(self, input_video_path, segments, output_video_path):
        """
        Extracts and combines specified segments from a video.

        Args:
            input_video_path (str): Path to the input video.
            segments (list of [start, end]): List of [start, end] times in seconds.
            output_video_path (str): Path to save the combined output video.
        """
        clips = []
        with VideoFileClip(input_video_path) as video:
            for start, end in segments:
                # Ensure end does not exceed video duration
                end = min(end, video.duration)
                if start < end:
                    clips.append(video.subclip(start, end))
            if clips:
                print("Combining segments...")
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(output_video_path, codec="libx264")
            else:
                print("No valid segments to combine.")


    def run(self, input_video_path: str, query: str, df, flag = "detections", output_video_path: str = None, output_csv_path: str = None, Retrieved_logs: str = None):
        
        if flag == "detections":
            # df = detect_objects(input_video_path, output_video_path, output_csv_path)

            system_prompt = self._prepare_system_prompt1()
            user_prompt = self._prepare_prompt1(self.df)

            generated_description_with_system, full_response_with_system = self.gpt_tool.run(input_video_path, system_prompt, user_prompt)

            
            # query = "Man jumping a wall"
            results = retrieve(query, generated_description_with_system)
            logs_timestamps = []
            print("\nTop matching logs:")
            for match in results:
                print(match)
                print(f"- {match['log']} (time: {match['time']}, score: {match['score']}%)")
                logs_timestamps.append(match['time'])

            self.Retrieved_logs = ""
            for match in results:
                self.Retrieved_logs += f"- {match['log']} (time: {match['time']}, score: {match['score']}%)\n"

            video_len = self.get_video_length(input_video_path)
            print("Video length in seconds:", video_len)
            segments = self.get_relevant_segments(logs_timestamps, video_len, padding=10)
            print("Extracted segments:", segments)
            self.extract_and_combine_segments(input_video_path, segments, "output_segments.mp4")

            return df, generated_description_with_system, self.Retrieved_logs
        

        elif flag == "logs":
            # df = detect_objects(input_video_path, output_video_path, output_csv_path)

            system_prompt = self._prepare_system_prompt2()
            user_prompt = self._prepare_prompt2(query, self.Retrieved_logs)

            generated_description_with_system, full_response_with_system = self.gpt_tool.run("output_segments.mp4", system_prompt, user_prompt)

            
            return None, generated_description_with_system, None