import logging
import time
import google.generativeai as genai
import matplotlib.pyplot as plt
from ikomia.dataprocess.workflow import Workflow
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import numpy as np
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import json
import re

class ChromaLogHandler:
    def __init__(self):
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.db = None

    def prepare_logs(self, logs):
        """Convert logs to plain text format for embedding, ensuring uniqueness."""

        seen = set()
        unique_logs = []
        for log in logs:
            entry = f"{log['activity']} at ``` timestamp {log['timestamp']} ``` "
            if entry not in seen:
                unique_logs.append(entry)
                seen.add(entry)
        return unique_logs


    def build_vector_store(self, logs):
        """Create Chroma vector DB from logs (in-memory)."""
        documents = self.prepare_logs(logs)
        metadatas = [{"time": log["timestamp"]} for log in logs]

        self.db = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas
        )

    def query_logs(self, query: str, k: int = 5):
        """Query the vector DB for similar logs."""
        if not self.db:
            raise RuntimeError("Vector store not built.")
        results = self.db.similarity_search_with_score(query, k=10)  # Fetch more to filter
        seen_logs = set()
        formatted_results = []
        for doc, score in results:
            log_text = doc.page_content
            if log_text not in seen_logs:
                match = re.search(r'timestamp (\d{2}:\d{2}:\d{2})', log_text)
                if match:
                    timestamp = match.group(1)
                    print("Timestamp:", timestamp)
                else:
                    print("No timestamp found.")
                    timestamp = None

                formatted_results.append({
                    "log": log_text,
                    "time": timestamp,
                    "score": round((1 - score) * 100, 2)
                })
                seen_logs.add(log_text)
            if len(formatted_results) == k:
                break
        return formatted_results



def wait_for_file_activation(file, timeout=60, interval=2):
    start_time = time.time()
    while time.time() - start_time < timeout:
        refreshed = genai.get_file(file.name)
        if refreshed.state.name == "ACTIVE":
            print(f"File {file.display_name} is ACTIVE and ready to use.")
            return refreshed
        elif refreshed.state.name == "FAILED":
            raise RuntimeError(f"File {file.display_name} failed to activate.")
        else:
            print(f"Waiting for file to become ACTIVE. Current state: {refreshed.state.name}")
            time.sleep(interval)
    raise TimeoutError(f"File {file.display_name} did not become ACTIVE in time.")


import json
import re

def retrieve(query, generated_description_with_system):

    print(generated_description_with_system)
    # Step 1: Extract content between any markdown code fence (with or without language identifier)
    code_pattern = re.compile(r'```(?:\w*)?\s*([\s\S]*?)\s*```')
    match = code_pattern.search(generated_description_with_system)
    logs = [{}]

    if match:
        extracted_content = match.group(1).strip()
        print("Extracted content:")
        print(extracted_content)

        # Step 2: Check if it looks like JSON (starts with { or [)
        if re.match(r'^\s*[\{\[]', extracted_content):
            # Step 3: Fix common JSON syntax issues
            # Remove trailing commas before closing brackets or braces
            cleaned_json = re.sub(r',(\s*[\}\]])', r'\1', extracted_content)

            # Step 4: Try to parse the JSON
            try:
                parsed_json = json.loads(cleaned_json)

                # Access data
                print("Logs/Activity:")
                for entry in parsed_json["Logs/Activity"]:
                    print(f"- {entry['timestamp']}: {entry['activity']}")
        
                logs = parsed_json["Logs/Activity"]

                print("\nAlerts:")
                for alert in parsed_json["Alerts"]:
                    print(f"- {alert['timestamp']}: {alert['alert']}")

                print("\nSuccessfully parsed JSON:")
                print(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError as e:
                print(f"\nFailed to parse JSON: {e}")
                print("Cleaned JSON (may require further fixing):")
                print(cleaned_json)
        else:
            print("Extracted content doesn't appear to be JSON")
    else:
        print("No content found between code fences")


    handler = ChromaLogHandler()

    # Build DB from logs
    handler.build_vector_store(logs)

    # Query
    results = handler.query_logs(query)

    return results

import threading
import queue
import cv2
import pandas as pd
import numpy as np
from ikomia.dataprocess.workflow import Workflow

# def detect_objects_parallel(input_video_path, output_video_path, skip_frames=5, output_csv_path="tracked_objects.csv"):
#     frame_queue = queue.Queue(maxsize=50)  # Limit to avoid memory overflow
#     stop_signal = object()
#     tracking_data = {}
#     debug_frames = 5

#     # Video writer and properties will be set after reading the first frame
#     video_writer = [None]
#     frame_width = [None]
#     frame_height = [None]
#     frame_rate = [None]
#     total_frames = [None]

#     def reader():
#         stream = cv2.VideoCapture(input_video_path)
#         frame_num = 0
#         # Get video properties
#         frame_width[0] = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height[0] = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         frame_rate[0] = stream.get(cv2.CAP_PROP_FPS)
#         total_frames[0] = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         video_writer[0] = cv2.VideoWriter(output_video_path, fourcc, frame_rate[0], (frame_width[0], frame_height[0]))

#         while True:
#             ret, frame = stream.read()
#             if not ret:
#                 break
#             frame_queue.put((frame_num, frame))
#             frame_num += 1
#         frame_queue.put(stop_signal)
#         stream.release()

#     def processor():
#         wf = Workflow()
#         # Add YOLOv11 detector
#         detector = wf.add_task(name="infer_yolo_v11", auto_connect=True)
#         detector.set_parameters({
#             "categories": "all",
#             "conf_thres": "0.3"
#         })
#         # Add DeepSORT tracker
#         tracking = wf.add_task(name="infer_deepsort", auto_connect=True)
#         tracking.set_parameters({
#             "categories": "all",
#             "conf_thres": "0.3",
#             "iou_thres": "0.5",
#             "max_age": "30",
#             "min_hits": "1",
#             "max_cosine_distance": "0.25"
#         })

#         while True:
#             item = frame_queue.get()
#             if item is stop_signal:
#                 break
#             frame_num, frame = item

#             # Optionally resize for speed (uncomment if needed)
#             # frame = cv2.resize(frame, (320, 480))

#             if frame_num % skip_frames != 0:
#                 continue

#             wf.run_on(array=frame)
#             image_out = tracking.get_output(0)
#             obj_detect_out = tracking.get_output(1)

#             # Debug output for first few frames
#             if frame_num < debug_frames:
#                 print(f"\n===== DEBUG INFO FOR FRAME {frame_num} =====")
#                 print("Object detection output type:", type(obj_detect_out))
#                 if hasattr(obj_detect_out, 'boxIDs'):
#                     print(f"boxIDs present: {len(obj_detect_out.boxIDs)} boxes")
#                 if hasattr(obj_detect_out, 'boxes'):
#                     print(f"boxes present: {len(obj_detect_out.boxes)} boxes")
#                 if hasattr(obj_detect_out, 'labels'):
#                     print(f"labels present: {obj_detect_out.labels}")

#             # Process detection results for the current frame
#             if obj_detect_out is not None:
#                 try:
#                     if hasattr(obj_detect_out, 'boxIDs'):
#                         box_ids = obj_detect_out.boxIDs
#                         boxes = obj_detect_out.boxes
#                         labels = obj_detect_out.labels

#                         if frame_num < debug_frames:
#                             print(f"Processing using boxIDs method: {len(box_ids)} objects")
#                             print(f"Labels: {labels}")

#                         for i, box_id in enumerate(box_ids):
#                             track_id = box_id
#                             class_name = labels[i] if i < len(labels) else "unknown"

#                             if track_id in tracking_data:
#                                 tracking_data[track_id]['end_frame'] = frame_num
#                                 tracking_data[track_id]['frames_visible'].append(frame_num)
#                             else:
#                                 tracking_data[track_id] = {
#                                     'class': class_name,
#                                     'start_frame': frame_num,
#                                     'end_frame': frame_num,
#                                     'frames_visible': [frame_num]
#                                 }
#                                 if frame_num < debug_frames:
#                                     print(f"New object detected: ID {track_id}, Class {class_name}")
#                     else:
#                         objects = []
#                         try:
#                             objects = obj_detect_out.get_objects()
#                         except:
#                             pass
#                         if not objects and hasattr(obj_detect_out, 'objects'):
#                             objects = obj_detect_out.objects
#                         if not objects and hasattr(obj_detect_out, 'items'):
#                             objects = obj_detect_out.items

#                         if frame_num < debug_frames:
#                             print(f"Processing using objects method: {len(objects)} objects")

#                         for obj in objects:
#                             if hasattr(obj, 'm_track_id'):
#                                 track_id = obj.m_track_id
#                             elif hasattr(obj, 'track_id'):
#                                 track_id = obj.track_id
#                             else:
#                                 track_id = id(obj)

#                             if hasattr(obj, 'm_label'):
#                                 class_name = obj.m_label
#                             elif hasattr(obj, 'class_name'):
#                                 class_name = obj.class_name
#                             elif hasattr(obj, 'label'):
#                                 class_name = obj.label
#                             else:
#                                 class_name = "unknown"

#                             if track_id in tracking_data:
#                                 tracking_data[track_id]['end_frame'] = frame_num
#                                 tracking_data[track_id]['frames_visible'].append(frame_num)
#                             else:
#                                 tracking_data[track_id] = {
#                                     'class': class_name,
#                                     'start_frame': frame_num,
#                                     'end_frame': frame_num,
#                                     'frames_visible': [frame_num]
#                                 }
#                                 if frame_num < debug_frames:
#                                     print(f"New object detected: ID {track_id}, Class {class_name}")
#                 except Exception as e:
#                     print(f"Error processing objects in frame {frame_num}: {str(e)}")

#             # Create a visual output with frame number
#             img_out = image_out.get_image_with_graphics(obj_detect_out)
#             img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

#             # Add frame counter to the image
#             cv2.putText(img_res, f"Frame: {frame_num}/{total_frames[0]}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Write to output video
#             if video_writer[0] is not None:
#                 video_writer[0].write(img_res)

#             # Print progress
#             if frame_num % 100 == 0:
#                 print(f"Processing frame {frame_num}/{total_frames[0]}")

#     t_reader = threading.Thread(target=reader)
#     t_processor = threading.Thread(target=processor)
#     t_reader.start()
#     t_processor.start()
#     t_reader.join()
#     t_processor.join()

#     # Release video writer
#     if video_writer[0] is not None:
#         video_writer[0].release()

#     # Post-processing (same as in your detect_objects)
#     print(f"\nTotal tracked objects: {len(tracking_data)}")
#     for track_id, data in list(tracking_data.items())[:5]:
#         print(f"Object ID {track_id}: Class {data['class']}, Frames: {data['start_frame']}-{data['end_frame']}")

#     min_frames_threshold = 1
#     filtered_tracking_data = {k: v for k, v in tracking_data.items()
#                               if len(v['frames_visible']) >= min_frames_threshold}

#     print(f"\nAfter filtering (min {min_frames_threshold} frames): {len(filtered_tracking_data)} objects")

#     if not filtered_tracking_data:
#         print("No objects were tracked. Creating empty CSV files.")
#         df = pd.DataFrame(columns=['object_id', 'class', 'start_frame', 'end_frame',
#                                    'frames_visible', 'total_frames', 'had_occlusion',
#                                    'visibility_percentage'])
#         simplified_df = pd.DataFrame(columns=['class', 'start_frame', 'end_frame'])
#     else:
#         for track_id, data in filtered_tracking_data.items():
#             frames_visible = data['frames_visible']
#             if len(frames_visible) >= 2:
#                 frame_diffs = np.diff(frames_visible)
#                 has_occlusion = np.any(frame_diffs > 1)
#                 visibility_percentage = len(frames_visible) / (data['end_frame'] - data['start_frame'] + 1) * 100
#                 data['had_occlusion'] = has_occlusion
#                 data['visibility_percentage'] = round(visibility_percentage, 2)

#         tracking_records = []
#         for track_id, data in filtered_tracking_data.items():
#             tracking_records.append({
#                 'object_id': track_id,
#                 'class': data['class'],
#                 'start_frame': data['start_frame'],
#                 'end_frame': data['end_frame'],
#                 'frames_visible': len(data['frames_visible']),
#                 'total_frames': data['end_frame'] - data['start_frame'] + 1,
#                 'had_occlusion': data.get('had_occlusion', False),
#                 'visibility_percentage': data.get('visibility_percentage', 100.0)
#             })

#         df = pd.DataFrame(tracking_records)
#         simplified_df = pd.DataFrame(tracking_records)[['class', 'start_frame', 'end_frame']]

#     if not df.empty:
#         df = df.sort_values(['class', 'start_frame'])
#     if not simplified_df.empty:
#         simplified_df = simplified_df.sort_values(['class', 'start_frame'])

#     if not df.empty:
#         print("\nDetected objects by class:")
#         class_counts = df['class'].value_counts()
#         for class_name, count in class_counts.items():
#             print(f"{class_name}: {count} instances")

#     df.to_csv(output_csv_path, index=False)
#     simplified_df.to_csv(output_csv_path.replace(".csv", "_simplified.csv"), index=False)

#     return df

import cv2
import threading
import queue
import pandas as pd
import numpy as np
from ikomia.dataprocess.workflow import Workflow


def process_video_with_debug(input_video_path, output_video_path, output_csv_path, skip_frames=10):
    """
    Process video with object detection and tracking, including comprehensive debugging
    """
    frame_queue = queue.Queue(maxsize=50)  # Limit to avoid memory overflow
    stop_signal = object()
    tracking_data = {}
    debug_frames = 5
    debug_interval = 100  # Debug every 100th frame

    # Video writer and properties will be set after reading the first frame
    video_writer = [None]
    frame_width = [None]
    frame_height = [None]
    frame_rate = [None]
    total_frames = [None]

    def reader():
        """Read video frames and put them in queue"""
        stream = cv2.VideoCapture(input_video_path)
        frame_num = 0
        
        # Get video properties
        frame_width[0] = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height[0] = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate[0] = stream.get(cv2.CAP_PROP_FPS)
        total_frames[0] = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width[0]}x{frame_height[0]}, {frame_rate[0]} FPS, {total_frames[0]} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer[0] = cv2.VideoWriter(output_video_path, fourcc, frame_rate[0], (frame_width[0], frame_height[0]))

        while True:
            ret, frame = stream.read()
            if not ret:
                break
            frame_queue.put((frame_num, frame))
            frame_num += 1
            
        frame_queue.put(stop_signal)
        stream.release()
        print(f"Reader finished. Total frames read: {frame_num}")

    def processor():
        """Process frames with object detection and tracking"""
        # Initialize workflow
        wf = Workflow()
        
        # Add YOLOv11 detector
        detector = wf.add_task(name="infer_yolo_v11", auto_connect=True)
        detector.set_parameters({
            "categories": "all",
            "conf_thres": "0.3"
        })
        
        # Add DeepSORT tracker
        tracking = wf.add_task(name="infer_deepsort", auto_connect=True)
        tracking.set_parameters({
            "categories": "all",
            "conf_thres": "0.3",
            "iou_thres": "0.5",
            "max_age": "30",
            "min_hits": "1",
            "max_cosine_distance": "0.25"
        })

        frames_processed = 0
        frames_with_objects = 0

        while True:
            item = frame_queue.get()
            if item is stop_signal:
                break
            
            frame_num, frame = item

            # Optionally resize for speed (uncomment if needed)
            # frame = cv2.resize(frame, (320, 480))

            # Skip frames if specified
            if frame_num % skip_frames != 0:
                continue

            frames_processed += 1

            # Run workflow on frame
            try:
                wf.run_on(array=frame)
                image_out = tracking.get_output(0)
                obj_detect_out = tracking.get_output(1)
            except Exception as e:
                print(f"ERROR running workflow on frame {frame_num}: {str(e)}")
                continue

            # Enhanced debug output
            should_debug = (frame_num < debug_frames) or (frame_num % debug_interval == 0)

            if should_debug:
                print(f"\n{'='*50}")
                print(f"DEBUG INFO FOR FRAME {frame_num}")
                print(f"{'='*50}")
                print(f"Frames processed so far: {frames_processed}")
                print(f"Object detection output type: {type(obj_detect_out)}")
                
                if obj_detect_out is not None:
                    print(f"Object detection output attributes: {[attr for attr in dir(obj_detect_out) if not attr.startswith('_')]}")
                    
                    # Check all possible attributes
                    attrs_to_check = ['boxIDs', 'boxes', 'labels', 'objects', 'items', 'detections', 'tracks', 'get_objects']
                    for attr in attrs_to_check:
                        if hasattr(obj_detect_out, attr):
                            try:
                                value = getattr(obj_detect_out, attr)
                                if callable(value):
                                    print(f"{attr}: <method>")
                                elif isinstance(value, (list, tuple, np.ndarray)):
                                    print(f"{attr}: {type(value)} with length {len(value)}")
                                    if len(value) > 0 and len(value) < 10:
                                        print(f"  Content: {value}")
                                else:
                                    print(f"{attr}: {type(value)} = {value}")
                            except Exception as e:
                                print(f"{attr}: Error accessing - {e}")
                else:
                    print("obj_detect_out is None!")

            # Process detection results for the current frame
            objects_found_this_frame = 0
            
            if obj_detect_out is not None:
                try:
                    # Method 1: Try boxIDs approach
                    method1_success = False
                    if hasattr(obj_detect_out, 'boxIDs') and obj_detect_out.boxIDs is not None:
                        try:
                            box_ids = obj_detect_out.boxIDs
                            boxes = getattr(obj_detect_out, 'boxes', [])
                            labels = getattr(obj_detect_out, 'labels', [])
                            
                            if should_debug:
                                print(f"\nMethod 1 - boxIDs approach:")
                                print(f"  boxIDs: {box_ids} (type: {type(box_ids)})")
                                print(f"  boxes: {len(boxes) if hasattr(boxes, '__len__') else 'N/A'}")
                                print(f"  labels: {labels}")

                            if hasattr(box_ids, '__len__') and len(box_ids) > 0:
                                method1_success = True
                                objects_found_this_frame = len(box_ids)
                                
                                for i, box_id in enumerate(box_ids):
                                    track_id = box_id
                                    class_name = labels[i] if i < len(labels) else "unknown"
                                    
                                    if should_debug:
                                        print(f"  Processing object {i+1}: ID={track_id}, Class={class_name}")

                                    if track_id in tracking_data:
                                        tracking_data[track_id]['end_frame'] = frame_num
                                        tracking_data[track_id]['frames_visible'].append(frame_num)
                                        if should_debug:
                                            print(f"    Updated existing track {track_id}")
                                    else:
                                        tracking_data[track_id] = {
                                            'class': class_name,
                                            'start_frame': frame_num,
                                            'end_frame': frame_num,
                                            'frames_visible': [frame_num]
                                        }
                                        if should_debug:
                                            print(f"    NEW OBJECT TRACKED: ID {track_id}, Class '{class_name}'")
                        except Exception as e:
                            print(f"Error in Method 1: {e}")
                    
                    # Method 2: Try objects/items approach if Method 1 didn't work
                    if not method1_success:
                        if should_debug:
                            print(f"\nMethod 2 - objects approach:")
                        
                        objects = []
                        
                        # Try different ways to get objects
                        try:
                            if hasattr(obj_detect_out, 'get_objects'):
                                objects = obj_detect_out.get_objects()
                                if should_debug:
                                    print(f"  get_objects() returned: {len(objects) if objects else 0} objects")
                        except Exception as e:
                            if should_debug:
                                print(f"  get_objects() failed: {e}")
                        
                        if not objects and hasattr(obj_detect_out, 'objects'):
                            objects = obj_detect_out.objects
                            if should_debug:
                                print(f"  .objects attribute: {len(objects) if objects else 0} objects")
                        
                        if not objects and hasattr(obj_detect_out, 'items'):
                            objects = obj_detect_out.items
                            if should_debug:
                                print(f"  .items attribute: {len(objects) if objects else 0} objects")
                        
                        if not objects and hasattr(obj_detect_out, 'detections'):
                            objects = obj_detect_out.detections
                            if should_debug:
                                print(f"  .detections attribute: {len(objects) if objects else 0} objects")
                        
                        if not objects and hasattr(obj_detect_out, 'tracks'):
                            objects = obj_detect_out.tracks
                            if should_debug:
                                print(f"  .tracks attribute: {len(objects) if objects else 0} objects")

                        if objects and len(objects) > 0:
                            objects_found_this_frame = len(objects)
                            
                            if should_debug:
                                print(f"  Processing {len(objects)} objects")
                                print(f"  First object type: {type(objects[0])}")
                                print(f"  First object attributes: {[attr for attr in dir(objects[0]) if not attr.startswith('_')]}")

                            for i, obj in enumerate(objects):
                                # Try to get track ID
                                track_id = None
                                possible_id_attrs = ['m_track_id', 'track_id', 'id', 'box_id', 'object_id']
                                for attr in possible_id_attrs:
                                    if hasattr(obj, attr):
                                        track_id = getattr(obj, attr)
                                        break
                                
                                if track_id is None:
                                    track_id = id(obj)  # Use object memory ID as fallback

                                # Try to get class name
                                class_name = "unknown"
                                possible_class_attrs = ['m_label', 'class_name', 'label', 'name', 'class']
                                for attr in possible_class_attrs:
                                    if hasattr(obj, attr):
                                        class_name = getattr(obj, attr)
                                        break

                                if should_debug:
                                    print(f"  Processing object {i+1}: ID={track_id}, Class={class_name}")

                                if track_id in tracking_data:
                                    tracking_data[track_id]['end_frame'] = frame_num
                                    tracking_data[track_id]['frames_visible'].append(frame_num)
                                    if should_debug:
                                        print(f"    Updated existing track {track_id}")
                                else:
                                    tracking_data[track_id] = {
                                        'class': class_name,
                                        'start_frame': frame_num,
                                        'end_frame': frame_num,
                                        'frames_visible': [frame_num]
                                    }
                                    if should_debug:
                                        print(f"    NEW OBJECT TRACKED: ID {track_id}, Class '{class_name}'")
                        elif should_debug:
                            print(f"  No objects found via Method 2")
                    
                    if objects_found_this_frame > 0:
                        frames_with_objects += 1
                    
                    if should_debug:
                        print(f"\nFrame {frame_num} summary:")
                        print(f"  Objects found this frame: {objects_found_this_frame}")
                        print(f"  Total tracking data entries: {len(tracking_data)}")
                        print(f"  Frames with objects so far: {frames_with_objects}/{frames_processed}")
                        
                except Exception as e:
                    print(f"ERROR processing objects in frame {frame_num}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

            # Create a visual output with frame number
            try:
                img_out = image_out.get_image_with_graphics(obj_detect_out)
                img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

                # Add frame counter to the image
                cv2.putText(img_res, f"Frame: {frame_num}/{total_frames[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add object count to the image
                cv2.putText(img_res, f"Objects: {objects_found_this_frame}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Write to output video
                if video_writer[0] is not None:
                    video_writer[0].write(img_res)
            except Exception as e:
                print(f"ERROR creating visual output for frame {frame_num}: {e}")

            # Print progress
            if frame_num % 100 == 0:
                print(f"Processing frame {frame_num}/{total_frames[0]} - Objects tracked: {len(tracking_data)}")

            # Regular tracking summary
            if frame_num % 500 == 0 and len(tracking_data) > 0:
                print(f"\n=== TRACKING SUMMARY AT FRAME {frame_num} ===")
                print(f"Total objects being tracked: {len(tracking_data)}")
                class_summary = {}
                for track_id, data in tracking_data.items():
                    class_name = data['class']
                    if class_name not in class_summary:
                        class_summary[class_name] = 0
                    class_summary[class_name] += 1
                
                for class_name, count in class_summary.items():
                    print(f"  {class_name}: {count} tracked objects")
                print("="*50)

        print(f"\nProcessor finished. Frames processed: {frames_processed}, Frames with objects: {frames_with_objects}")

    # Start threads
    t_reader = threading.Thread(target=reader)
    t_processor = threading.Thread(target=processor)
    t_reader.start()
    t_processor.start()
    t_reader.join()
    t_processor.join()

    # Release video writer
    if video_writer[0] is not None:
        video_writer[0].release()

    # Post-processing (create DataFrame and save results)
    print(f"\n{'='*60}")
    print("POST-PROCESSING RESULTS")
    print(f"{'='*60}")
    print(f"Total tracked objects: {len(tracking_data)}")
    
    if len(tracking_data) > 0:
        print("\nFirst 5 tracked objects:")
        for track_id, data in list(tracking_data.items())[:5]:
            print(f"  Object ID {track_id}: Class '{data['class']}', Frames: {data['start_frame']}-{data['end_frame']}")
    else:
        print("WARNING: No objects were tracked!")

    # Apply minimum frames threshold
    min_frames_threshold = 1
    filtered_tracking_data = {k: v for k, v in tracking_data.items()
                              if len(v['frames_visible']) >= min_frames_threshold}

    print(f"\nAfter filtering (min {min_frames_threshold} frames): {len(filtered_tracking_data)} objects")

    if not filtered_tracking_data:
        print("No objects were tracked. Creating empty CSV files.")
        df = pd.DataFrame(columns=['object_id', 'class', 'start_frame', 'end_frame',
                                   'frames_visible', 'total_frames', 'had_occlusion',
                                   'visibility_percentage'])
        simplified_df = pd.DataFrame(columns=['class', 'start_frame', 'end_frame'])
    else:
        # Calculate occlusion and visibility data
        for track_id, data in filtered_tracking_data.items():
            frames_visible = data['frames_visible']
            if len(frames_visible) >= 2:
                frame_diffs = np.diff(frames_visible)
                has_occlusion = np.any(frame_diffs > 1)
                visibility_percentage = len(frames_visible) / (data['end_frame'] - data['start_frame'] + 1) * 100
                data['had_occlusion'] = has_occlusion
                data['visibility_percentage'] = round(visibility_percentage, 2)
            else:
                data['had_occlusion'] = False
                data['visibility_percentage'] = 100.0

        # Create tracking records
        tracking_records = []
        for track_id, data in filtered_tracking_data.items():
            tracking_records.append({
                'object_id': track_id,
                'class': data['class'],
                'start_frame': data['start_frame'],
                'end_frame': data['end_frame'],
                'frames_visible': len(data['frames_visible']),
                'total_frames': data['end_frame'] - data['start_frame'] + 1,
                'had_occlusion': data.get('had_occlusion', False),
                'visibility_percentage': data.get('visibility_percentage', 100.0)
            })

        df = pd.DataFrame(tracking_records)
        simplified_df = pd.DataFrame(tracking_records)[['class', 'start_frame', 'end_frame']]

    # Sort DataFrames
    if not df.empty:
        df = df.sort_values(['class', 'start_frame'])
    if not simplified_df.empty:
        simplified_df = simplified_df.sort_values(['class', 'start_frame'])

    # Print class summary
    if not df.empty:
        print("\nDetected objects by class:")
        class_counts = df['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} instances")
    else:
        print("\nNo objects detected in final DataFrame!")

    # Save CSV files
    df.to_csv(output_csv_path, index=False)
    simplified_df.to_csv(output_csv_path.replace(".csv", "_simplified.csv"), index=False)
    
    print(f"\nResults saved to:")
    print(f"  Full CSV: {output_csv_path}")
    print(f"  Simplified CSV: {output_csv_path.replace('.csv', '_simplified.csv')}")
    print(f"  Output video: {output_video_path}")

    return df


def detect_objects(input_video_path, output_video_path, skip_frames = 5, output_csv_path = "tracked_objects.csv"):
    # Initialize workflow
    wf = Workflow()

    # Add YOLOv11 detector (make sure it exists in Ikomia)
    detector = wf.add_task(name="infer_yolo_v11", auto_connect=True)
    # Ensure we're detecting all COCO classes, not just persons
    detector.set_parameters({
        "categories": "all",  # Detect all classes, not just persons
        "conf_thres": "0.3"   # Lower confidence threshold to catch more objects
    })

    # Add DeepSORT tracker with parameters to handle occlusion better
    tracking = wf.add_task(name="infer_deepsort", auto_connect=True)
    tracking.set_parameters({
        "categories": "all",          # Track all detected classes
        "conf_thres": "0.3",          # Lower confidence threshold
        "iou_thres": "0.5",           # IoU threshold for NMS
        "max_age": "30",              # Number of frames to keep tracking lost objects
        "min_hits": "1",              # Reduced min hits to start tracking
        "max_cosine_distance": "0.25" # Slightly increased to allow more flexible matching
    })

    # Read video
    stream = cv2.VideoCapture(input_video_path)
    if not stream.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = stream.get(cv2.CAP_PROP_FPS)
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Dictionary to track objects
    # Format: {track_id: {'class': class_name, 'start_frame': frame_num, 'end_frame': frame_num, 'frames_visible': [frame_nums]}}
    tracking_data = {}

    # Debug flag to print detailed info for first few frames
    debug_frames = 5

    # Process video frame by frame
    frame_num = 0

    try:
        while True:
            ret, frame = stream.read()
            frame = cv2.resize(frame, (320, 480))
            if not ret:
                break

            # Run workflow
            wf.run_on(array=frame)
            image_out = tracking.get_output(0)
            obj_detect_out = tracking.get_output(1)

            if frame_num % skip_frames != 0:
                frame_num += 1
                continue

            # Debug output for first few frames
            if frame_num < debug_frames:
                print(f"\n===== DEBUG INFO FOR FRAME {frame_num} =====")
                print("Object detection output type:", type(obj_detect_out))
                if hasattr(obj_detect_out, 'boxIDs'):
                    print(f"boxIDs present: {len(obj_detect_out.boxIDs)} boxes")
                if hasattr(obj_detect_out, 'boxes'):
                    print(f"boxes present: {len(obj_detect_out.boxes)} boxes")
                if hasattr(obj_detect_out, 'labels'):
                    print(f"labels present: {obj_detect_out.labels}")

            # Process detection results for the current frame
            if obj_detect_out is not None:
                try:
                    # Check which attributes are available for accessing objects
                    if hasattr(obj_detect_out, 'boxIDs'):
                        # Method for accessing boxIDs, boxes, labels directly
                        box_ids = obj_detect_out.boxIDs
                        boxes = obj_detect_out.boxes
                        labels = obj_detect_out.labels

                        if frame_num < debug_frames:
                            print(f"Processing using boxIDs method: {len(box_ids)} objects")
                            print(f"Labels: {labels}")

                        for i, box_id in enumerate(box_ids):
                            track_id = box_id
                            class_name = labels[i] if i < len(labels) else "unknown"

                            # Don't rename classes to keep original labels for all objects
                            # Just use the raw class name as detected

                            # Update tracking data
                            if track_id in tracking_data:
                                # Object already being tracked, update end frame
                                tracking_data[track_id]['end_frame'] = frame_num
                                tracking_data[track_id]['frames_visible'].append(frame_num)
                            else:
                                # New object detected
                                tracking_data[track_id] = {
                                    'class': class_name,
                                    'start_frame': frame_num,
                                    'end_frame': frame_num,
                                    'frames_visible': [frame_num]
                                }
                                if frame_num < debug_frames:
                                    print(f"New object detected: ID {track_id}, Class {class_name}")
                    else:
                        # Try to get objects through different methods
                        objects = []

                        # Method 1: Try get_objects()
                        try:
                            objects = obj_detect_out.get_objects()
                        except:
                            pass

                        # Method 2: Try direct object attribute access
                        if not objects and hasattr(obj_detect_out, 'objects'):
                            objects = obj_detect_out.objects

                        # Method 3: Try items attribute
                        if not objects and hasattr(obj_detect_out, 'items'):
                            objects = obj_detect_out.items

                        if frame_num < debug_frames:
                            print(f"Processing using objects method: {len(objects)} objects")

                        for obj in objects:
                            # Try different attribute names that might be present
                            if hasattr(obj, 'm_track_id'):
                                track_id = obj.m_track_id
                            elif hasattr(obj, 'track_id'):
                                track_id = obj.track_id
                            else:
                                # Generate a unique ID if track_id is not available
                                track_id = id(obj)

                            if hasattr(obj, 'm_label'):
                                class_name = obj.m_label
                            elif hasattr(obj, 'class_name'):
                                class_name = obj.class_name
                            elif hasattr(obj, 'label'):
                                class_name = obj.label
                            else:
                                class_name = "unknown"

                            # Don't rename classes - keep original class names

                            # Update tracking data
                            if track_id in tracking_data:
                                # Object already being tracked, update end frame
                                tracking_data[track_id]['end_frame'] = frame_num
                                tracking_data[track_id]['frames_visible'].append(frame_num)
                            else:
                                # New object detected
                                tracking_data[track_id] = {
                                    'class': class_name,
                                    'start_frame': frame_num,
                                    'end_frame': frame_num,
                                    'frames_visible': [frame_num]
                                }
                                if frame_num < debug_frames:
                                    print(f"New object detected: ID {track_id}, Class {class_name}")
                except Exception as e:
                    print(f"Error processing objects in frame {frame_num}: {str(e)}")

            # Create a visual output with frame number
            img_out = image_out.get_image_with_graphics(obj_detect_out)
            img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

            # Add frame counter to the image
            cv2.putText(img_res, f"Frame: {frame_num}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(img_res)

            # Display in Colab (uncomment for visualization)
            # if frame_num % 10 == 0:  # Show every 10th frame to reduce overhead
            #     plt.figure(figsize=(12, 8))
            #     plt.axis('off')
            #     plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            #     plt.title(f"Frame: {frame_num}/{total_frames}")
            #     clear_output(wait=True)
            #     plt.show()

            # Print progress
            if frame_num % 100 == 0:
                print(f"Processing frame {frame_num}/{total_frames}")

            frame_num += 1

    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
    finally:
        # Release resources
        stream.release()
        out.release()
        print(f"Completed processing {frame_num} frames")



    # Debug final tracking data
    print(f"\nTotal tracked objects: {len(tracking_data)}")
    for track_id, data in list(tracking_data.items())[:5]:  # Print first 5 objects for debug
        print(f"Object ID {track_id}: Class {data['class']}, Frames: {data['start_frame']}-{data['end_frame']}")

    # Filter out objects that appear in too few frames (likely false positives)
    min_frames_threshold = 5  # Objects must appear in at least this many frames
    filtered_tracking_data = {k: v for k, v in tracking_data.items()
                            if len(v['frames_visible']) >= min_frames_threshold}

    print(f"\nAfter filtering (min {min_frames_threshold} frames): {len(filtered_tracking_data)} objects")

    # Check if any objects were tracked
    if not filtered_tracking_data:
        print("No objects were tracked. Creating empty CSV files.")
        # Create empty DataFrames
        df = pd.DataFrame(columns=['object_id', 'class', 'start_frame', 'end_frame',
                                    'frames_visible', 'total_frames', 'had_occlusion',
                                    'visibility_percentage'])
        simplified_df = pd.DataFrame(columns=['class', 'start_frame', 'end_frame'])
    else:
        # Post-process tracking data to handle occlusion gaps
        for track_id, data in filtered_tracking_data.items():
            frames_visible = data['frames_visible']

            # Check for gaps in visibility (occlusion)
            if len(frames_visible) >= 2:
                frame_diffs = np.diff(frames_visible)
                has_occlusion = np.any(frame_diffs > 1)

                # Calculate actual visibility percentage
                visibility_percentage = len(frames_visible) / (data['end_frame'] - data['start_frame'] + 1) * 100

                # Add occlusion info to the data
                data['had_occlusion'] = has_occlusion
                data['visibility_percentage'] = round(visibility_percentage, 2)

        # Create DataFrame and save to CSV
        tracking_records = []
        for track_id, data in filtered_tracking_data.items():
            tracking_records.append({
                'object_id': track_id,
                'class': data['class'],
                'start_frame': data['start_frame'],
                'end_frame': data['end_frame'],
                'frames_visible': len(data['frames_visible']),
                'total_frames': data['end_frame'] - data['start_frame'] + 1,
                'had_occlusion': data.get('had_occlusion', False),
                'visibility_percentage': data.get('visibility_percentage', 100.0)
            })

        df = pd.DataFrame(tracking_records)

        # Create simplified DataFrame - including all object classes, not just persons
        simplified_df = pd.DataFrame(tracking_records)[['class', 'start_frame', 'end_frame']]

    # Sort DataFrames if they're not empty
    if not df.empty:
        df = df.sort_values(['class', 'start_frame'])
    if not simplified_df.empty:
        simplified_df = simplified_df.sort_values(['class', 'start_frame'])

    # Print data summary by class
    if not df.empty:
        print("\nDetected objects by class:")
        class_counts = df['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} instances")

    df.to_csv(output_csv_path, index=False)
    simplified_df.to_csv(output_csv_path.replace(".csv", "_simplified.csv"), index=False)

    return df
