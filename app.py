# import streamlit as st
# from my_agent import MyAgent
# from config import gemini_config

# # Set Streamlit page config
# st.set_page_config(page_title="Drone Security Analyst", layout="wide")

# # Initialize session state
# if "agent" not in st.session_state:
#     st.session_state.agent = None
# if "logs" not in st.session_state:
#     st.session_state.logs = None
# if "video_processed" not in st.session_state:
#     st.session_state.video_processed = False
# if "csv_path" not in st.session_state:
#     st.session_state.csv_path = None
# if "output_video" not in st.session_state:
#     st.session_state.output_video = None

# st.title("🚨 Drone Security Analyst Agent")

# st.sidebar.header("⚙️ Configuration")

# video_file = st.sidebar.text_input("Video Path", "combined.mp4")
# query = st.sidebar.text_input("Detection Query", "railway track")
# output_video_path = st.sidebar.text_input("Output Video", "output.mp4")
# output_csv_path = st.sidebar.text_input("Output CSV", "detections.csv")

# run_detection = st.sidebar.button("🔍 Run Detection")

# # Reset button
# if st.sidebar.button("🔄 Reset"):
#     st.session_state.agent = None
#     st.session_state.logs = None
#     st.session_state.video_processed = False
#     st.session_state.csv_path = None
#     st.session_state.output_video = None
#     st.success("Session reset!")

# # Step 1: Run detection and logging
# if run_detection:
#     st.session_state.agent = MyAgent()
#     st.session_state.agent.setup(gemini_config)

#     with st.spinner(f"Processing video: {video_file}"):
#         df, generated_logs, logs = st.session_state.agent.run(
#             input_video_path=video_file,
#             query=query,
#             flag="detections",
#             output_video_path=output_video_path,
#             output_csv_path=output_csv_path
#         )
#         st.session_state.logs = logs
#         st.session_state.video_processed = True
#         st.session_state.csv_path = output_csv_path
#         st.session_state.output_video = output_video_path
#         st.success("✅ Detection completed and logs generated!")

#     with st.expander("📜 Detection Summary"):
#         st.write(generated_logs)

#     with st.expander("📂 Detection CSV"):
#         st.dataframe(df)

# # Step 2: Ask follow-up questions
# if st.session_state.video_processed:
#     st.subheader("🤖 Ask a Question Based on the Logs")
#     ask_query = st.text_input("Ask anything (e.g., how many vehicles?)")
#     if st.button("🔎 Get Answer"):
#         if ask_query and st.session_state.logs:
#             _, answer, _ = st.session_state.agent.run(
#                 input_video_path=video_file,
#                 query=ask_query,
#                 flag="logs",
#                 Retrieved_logs=st.session_state.logs
#             )
#             st.success("✅ Answer generated:")
#             st.write(answer)
#         else:
#             st.warning("❗ Please run detection first or ask a valid question.")

# # Show output video (if available)
# if st.session_state.output_video:
#     st.subheader("📼 Output Video with Detections")
#     try:
#         with open(st.session_state.output_video, "rb") as f:
#             st.video(f.read())
#     except Exception as e:
#         st.warning(f"Could not load video: {e}")


import streamlit as st
from tools.OrchestratorTool import OrchestratorTool
from config import gemini_config
import pandas as pd
from utils.utils import process_video_with_debug

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🎥 Drone Surveillance Analyzer", layout="wide")
st.title("🛡️ Drone Surveillance Analyzer")

# --- Session State Initialization ---
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "df" not in st.session_state:
    st.session_state.df = None
if "video_path" not in st.session_state:
    st.session_state.video_path = ""
if "retrieved_logs" not in st.session_state:
    st.session_state.retrieved_logs = None

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Input Configuration")
video_path = st.sidebar.text_input("📁 Video Path", "./Videos/combined.mp4")
output_csv = st.sidebar.text_input("📄 Output CSV", "output.csv")

run_detection = st.sidebar.button("🔍 Run Object Detection")

# --- Reset Session Button ---
if st.sidebar.button("🔄 Reset"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# --- Run Detection and Initialize Orchestrator ---
if run_detection:
    with st.spinner("Running object detection and preparing analysis..."):
        # df = process_video_with_debug(video_path, "output_video", skip_frames= 15, output_csv_path=output_csv)
        df = pd.read_csv(r"output.csv")
        st.session_state.df = df
        st.session_state.video_path = video_path
        st.session_state.orchestrator = OrchestratorTool(gemini_config, df)
        st.success("✅ Detection complete. You can now ask queries below.")

# --- Query Section ---
if st.session_state.orchestrator:
    st.subheader("🤖 Ask a Surveillance Query")

    query_input = st.text_input("Type your query (e.g., count people, detect threats):", value="Count the number of people")
    if st.button("📊 Analyze"):
        with st.spinner("Analyzing with appropriate agent..."):
            description, logs, agent = st.session_state.orchestrator.run(
                st.session_state.video_path, "", query_input
            )
            st.session_state.retrieved_logs = logs
            st.success(f"✅ Analysis complete using `{agent}` agent.")
            st.markdown("**📝 Response:**")
            st.write(description)

# --- Show Output CSV ---
if st.session_state.df is not None:
    with st.expander("📂 Show Detection CSV"):
        st.dataframe(st.session_state.df)

# --- Show Video (if available) ---
if st.session_state.video_path:
    st.subheader("📼 Input Video")
    try:
        with open(st.session_state.video_path, "rb") as f:
            st.video(f.read())
    except Exception as e:
        st.warning(f"Could not load video: {e}")
