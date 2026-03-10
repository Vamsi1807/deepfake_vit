import json
from pathlib import Path

import streamlit as st

from inference_visualizer import process_and_annotate_video


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def format_timestamp(seconds):
    total = int(max(seconds, 0))
    mins = total // 60
    secs = total % 60
    return f"{mins:02d}:{secs:02d}"


st.set_page_config(page_title="Deepfake Video Analyzer", layout="wide")
st.title("Deepfake Video Analyzer")
st.caption("Upload a video, run inference, and inspect detected fake segments with reason signals.")

uploaded_file = st.file_uploader(
    "Upload input video",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

if uploaded_file is not None:
    input_path = UPLOAD_DIR / uploaded_file.name
    input_path.write_bytes(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running inference. This can take some time..."):
            result = process_and_annotate_video(str(input_path))

        if not result:
            st.error("Inference failed: no frames extracted from the uploaded video.")
            st.stop()

        output_video_path_value = result.get("annotated_video_path")
        analysis_json_path = Path(result["analysis_json_path"])
        output_video_path = Path(output_video_path_value) if output_video_path_value else None

        st.subheader("Annotated Output")
        if output_video_path is not None and output_video_path.exists():
            video_bytes = output_video_path.read_bytes()
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                "Download Annotated Video",
                data=video_bytes,
                file_name=output_video_path.name,
                mime="video/mp4",
            )
        else:
            if result.get("status") == "no_face_detected":
                st.warning("No face detected in the video. Processing stopped.")
            else:
                st.warning("Annotated video file was not found.")

        st.subheader("Fake Segment Analysis")
        if analysis_json_path.exists():
            analysis = json.loads(analysis_json_path.read_text(encoding="utf-8"))
            video_verdict = analysis.get("video_verdict", "UNKNOWN")
            fake_segments = analysis.get("fake_segments", [])

            st.write(f"Video verdict: `{video_verdict}`")
            if video_verdict == "NO_FACE_DETECTED":
                st.info("Result: No face detected.")

            if not fake_segments:
                st.info("No fake segments detected.")
            else:
                st.write(f"Detected {len(fake_segments)} fake segment(s).")
                for idx, segment in enumerate(fake_segments, start=1):
                    start_time = format_timestamp(segment.get("start_time_sec", 0))
                    end_time = format_timestamp(segment.get("end_time_sec", 0))
                    peak_score = segment.get("peak_fake_score", 0.0)
                    face_id = segment.get("face_id", -1)
                    st.markdown(
                        f"**Segment {idx}** | Face `{face_id}` | `{start_time}` to `{end_time}` | Peak score `{peak_score:.3f}`"
                    )
                    for reason in segment.get("reason_signals", []):
                        st.write(f"- {reason}")

            with st.expander("Raw analysis JSON"):
                st.json(analysis)
        else:
            st.warning("Analysis JSON file was not found.")

else:
    st.info("Upload a video to begin.")
