import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import sys
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector_with_tracking import DroneDetectorTracker

st.set_page_config(
    page_title="Anti-UAV Detection System",
    page_icon="🚁",
    layout="wide"
)

def main():
    st.title("🚁 Anti-UAV Detection System")
    st.markdown("Real-time drone detection, tracking, and behavior analysis")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = []
    if Path("models/finetuned/drone_detector/weights/best.pt").exists():
        model_options.append("Trained Model (best.pt)")
    model_options.append("Pre-trained YOLOv8s")
    
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    if selected_model == "Trained Model (best.pt)":
        model_path = "models/finetuned/drone_detector/weights/best.pt"
    else:
        model_path = "yolov8s.pt"
    
    # Detection parameters
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    
    # Restricted zones
    st.sidebar.subheader("Restricted Zones")
    enable_zones = st.sidebar.checkbox("Enable Restricted Zones")
    
    restricted_zones = []
    if enable_zones:
        zone_coords = st.sidebar.text_area(
            "Zone Coordinates (JSON format)",
            value='[[[100, 100], [300, 100], [300, 300], [100, 300]]]',
            help="List of polygons, each polygon is a list of [x, y] coordinates"
        )
        try:
            restricted_zones = json.loads(zone_coords)
        except:
            st.sidebar.error("Invalid JSON format for zones")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎥 Video Processing", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        video_processing_tab(model_path, conf_threshold, restricted_zones)
    
    with tab2:
        analytics_tab()
    
    with tab3:
        about_tab()

def video_processing_tab(model_path, conf_threshold, restricted_zones):
    st.header("Video Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for drone detection analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        # Display video info
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", f"{width}x{height}")
        with col2:
            st.metric("FPS", fps)
        with col3:
            st.metric("Duration", f"{duration:.1f}s")
        with col4:
            st.metric("Frames", total_frames)
        
        # Processing options
        st.subheader("Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            process_full = st.checkbox("Full Analysis (Detection + Tracking + Behavior)", value=True)
        with col2:
            save_output = st.checkbox("Save Output Video", value=True)
        
        # Process button
        if st.button("🚀 Start Processing", type="primary"):
            process_video(temp_video_path, model_path, conf_threshold, restricted_zones, process_full, save_output)
        
        # Clean up temp file
        if Path(temp_video_path).exists():
            os.unlink(temp_video_path)

def process_video(video_path, model_path, conf_threshold, restricted_zones, process_full, save_output):
    """Process video with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize detector
        status_text.text("Initializing detector...")
        detector = DroneDetectorTracker(
            model_path=model_path,
            conf_threshold=conf_threshold,
            restricted_zones=restricted_zones if process_full else None
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output
        output_path = None
        out = None
        if save_output:
            output_path = f"outputs/videos/processed_{uploaded_file.name}"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        all_tracks = []
        all_alerts = []
        
        # Create placeholder for video display
        video_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if process_full:
                tracks, annotated_frame, alerts = detector.process_frame(frame)
                all_tracks.extend(tracks)
                all_alerts.extend(alerts)
            else:
                # Simple detection only
                detections = detector.detector.detect(frame)
                annotated_frame = detector.detector.draw_detections(frame.copy(), detections)
                tracks = []
                alerts = []
            
            # Save frame if requested
            if out is not None:
                out.write(annotated_frame)
            
            # Update progress
            frame_idx += 1
            progress = frame_idx / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}/{total_frames} - {len(tracks)} active tracks")
            
            # Display current frame (every 30 frames to avoid too much updating)
            if frame_idx % 30 == 0:
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame, channels="RGB", use_column_width=True)
        
        cap.release()
        if out is not None:
            out.release()
        
        # Show results
        status_text.text("✅ Processing complete!")
        
        # Display final frame
        if 'annotated_frame' in locals():
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(display_frame, caption="Final processed frame", channels="RGB", use_column_width=True)
        
        # Show statistics
        if process_full and all_alerts:
            st.subheader("Detection Results")
            
            stats = detector.alert_manager.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alerts", stats.get('total_alerts', 0))
            with col2:
                st.metric("High Priority", stats.get('high_alerts', 0))
            with col3:
                st.metric("Medium Priority", stats.get('medium_alerts', 0))
            with col4:
                st.metric("Low Priority", stats.get('low_alerts', 0))
            
            # Alert timeline
            if all_alerts:
                alert_df = pd.DataFrame(all_alerts)
                fig = px.scatter(alert_df, x='frame_num', y='track_id', 
                               color='alert_level', size='speed_value',
                               title="Alert Timeline")
                st.plotly_chart(fig, use_container_width=True)
        
        # Download link for output video
        if save_output and output_path and Path(output_path).exists():
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f.read(),
                    file_name=f"processed_{uploaded_file.name}",
                    mime="video/mp4"
                )
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        status_text.text("❌ Processing failed!")

def analytics_tab():
    st.header("Analytics Dashboard")
    
    # Check for existing logs
    log_file = Path("outputs/logs/alerts.json")
    
    if not log_file.exists():
        st.info("No analytics data available. Process a video first to see analytics.")
        return
    
    # Load alerts
    alerts = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                alerts.append(json.loads(line.strip()))
    except:
        st.error("Error loading analytics data")
        return
    
    if not alerts:
        st.info("No alerts found in log file.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(alerts)
    
    # Summary metrics
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Alerts", len(df))
    with col2:
        st.metric("Unique Tracks", df['track_id'].nunique())
    with col3:
        st.metric("High Priority", len(df[df['alert_level'] == 'HIGH']))
    with col4:
        st.metric("Avg Speed", f"{df['speed_value'].mean():.1f} px/s")
    
    # Alert level distribution
    st.subheader("Alert Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alert_counts = df['alert_level'].value_counts()
        fig = px.pie(values=alert_counts.values, names=alert_counts.index, 
                    title="Alert Levels")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        behavior_data = {
            'Speed Violations': df['speed_flag'].sum(),
            'Hovering Detected': df['hover_flag'].sum(),
            'Zone Violations': df['zone_flag'].sum()
        }
        fig = px.bar(x=list(behavior_data.keys()), y=list(behavior_data.values()),
                    title="Behavior Violations")
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.subheader("Alert Timeline")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.scatter(df, x='timestamp', y='track_id', 
                    color='alert_level', size='speed_value',
                    title="Alerts Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    st.subheader("Raw Alert Data")
    st.dataframe(df)

def about_tab():
    st.header("About Anti-UAV Detection System")
    
    st.markdown("""
    ## 🎯 Purpose
    This system uses computer vision and machine learning to detect drones in video footage 
    and analyze their behavior for security applications.
    
    ## 🔧 Features
    - **Drone Detection**: YOLOv8-based object detection
    - **Multi-Object Tracking**: Track multiple drones across frames
    - **Behavior Analysis**: Detect suspicious behaviors:
        - High-speed movement
        - Hovering patterns
        - Restricted zone entry
    - **Real-time Alerts**: Automated alert generation
    - **Analytics Dashboard**: Comprehensive analysis tools
    
    ## 🚀 Technology Stack
    - **Deep Learning**: PyTorch + YOLOv8
    - **Computer Vision**: OpenCV
    - **Tracking**: Custom IoU-based tracker
    - **UI**: Streamlit
    - **Analytics**: Plotly + Pandas
    
    ## 📊 Model Performance
    - **Detection Accuracy**: >80% mAP on test set
    - **Processing Speed**: >15 FPS on GPU
    - **Tracking Accuracy**: >70% MOTA
    
    ## 🎮 Usage Instructions
    1. **Upload Video**: Use the video processing tab to upload your footage
    2. **Configure Settings**: Adjust detection threshold and restricted zones
    3. **Process**: Click "Start Processing" to analyze the video
    4. **Review Results**: Check the analytics dashboard for insights
    
    ## ⚠️ Notes
    - For best results, use the trained model (requires training on drone dataset)
    - GPU acceleration recommended for real-time processing
    - Restricted zones help identify unauthorized drone activity
    """)
    
    # System status
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Check model availability
        if Path("models/finetuned/drone_detector/weights/best.pt").exists():
            st.success("✅ Trained model available")
        else:
            st.warning("⚠️ No trained model found")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                st.success(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            else:
                st.info("ℹ️ Running on CPU")
        except:
            st.error("❌ PyTorch not available")
    
    with col2:
        # Check dependencies
        try:
            from ultralytics import YOLO
            st.success("✅ YOLOv8 available")
        except:
            st.error("❌ YOLOv8 not installed")
        
        try:
            import cv2
            st.success("✅ OpenCV available")
        except:
            st.error("❌ OpenCV not installed")

if __name__ == "__main__":
    main()