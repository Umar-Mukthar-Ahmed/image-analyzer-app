"""
Streamlit Web UI for Image Analyzer App
Beautiful, interactive web interface
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import io
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.analyzer import ImageAnalyzer, AnalysisError
from src.decision.confidence_engine import ConfidenceEngine
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="Image Analyzer Pro",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    try:
        st.session_state.analyzer = ImageAnalyzer()
        st.session_state.confidence_engine = ConfidenceEngine()
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")
        st.stop()

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
        return "#4CAF50"  # Green
    elif confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def create_confidence_chart(items, name_attr='name', conf_attr='confidence'):
    """Create confidence bar chart"""
    if not items:
        return None

    # Prepare data
    names = [getattr(item, name_attr) for item in items[:10]]
    confidences = [getattr(item, conf_attr) * 100 for item in items[:10]]
    colors = [get_confidence_color(getattr(item, conf_attr)) for item in items[:10]]

    # Create chart
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=names,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{c:.1f}%" for c in confidences],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=max(300, len(names) * 40),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def create_reliability_gauge(reliability_score):
    """Create reliability gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=reliability_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Reliability"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': get_confidence_color(reliability_score)},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    fig.update_layout(height=300)

    return fig


def display_analysis_results(result, image):
    """Display complete analysis results"""

    # Get summary
    summary = st.session_state.analyzer.get_analysis_summary(result)

    # Main metrics
    st.markdown("### 📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Descriptions",
            len(result.descriptions),
            delta=None
        )

    with col2:
        st.metric(
            "Objects",
            len(result.objects),
            delta=None
        )

    with col3:
        st.metric(
            "Tags",
            len(result.tags),
            delta=None
        )

    with col4:
        st.metric(
            "Text Lines",
            len(result.ocr_text),
            delta=None
        )

    # Reliability Gauge
    st.markdown("### 🎯 Reliability Score")
    reliability = summary['reliability']

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_gauge = create_reliability_gauge(reliability['reliability_score'])
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown(f"""
        **Reliability Level:** {reliability['reliability_level']}

        **Metrics:**
        - Average Confidence: {reliability['average_confidence'] * 100:.1f}%
        - Total Features: {reliability['total_features']}
        - High Confidence: {reliability['high_confidence_features']}
        - Low Confidence: {reliability['low_confidence_features']}
        """)

    # Warnings
    if summary['warnings']:
        st.markdown("### ⚠️ Uncertainty Warnings")
        for warning in summary['warnings']:
            st.warning(warning)

    # Descriptions
    if result.has_descriptions:
        st.markdown("### 💬 Image Descriptions")
        for i, desc in enumerate(result.descriptions[:3], 1):
            confidence_pct = desc.confidence_percentage
            color = get_confidence_color(desc.confidence)

            st.markdown(f"""
            <div class="metric-card">
                <strong>{i}. {desc.text}</strong><br>
                <span style="color: {color};">Confidence: {confidence_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # Tags with chart
    if result.tags:
        st.markdown("### 🏷️ Tags")

        col1, col2 = st.columns([1, 1])

        with col1:
            for tag in result.tags[:10]:
                confidence_pct = tag.confidence_percentage
                color = get_confidence_color(tag.confidence)
                st.markdown(f"• **{tag.name}** - <span style='color: {color};'>{confidence_pct:.1f}%</span>",
                            unsafe_allow_html=True)

        with col2:
            fig_tags = create_confidence_chart(result.tags, 'name', 'confidence')
            if fig_tags:
                st.plotly_chart(fig_tags, use_container_width=True)

    # Objects
    if result.has_objects:
        st.markdown("### 📦 Detected Objects")

        col1, col2 = st.columns([1, 1])

        with col1:
            for obj in result.objects:
                confidence_pct = obj.confidence_percentage
                color = get_confidence_color(obj.confidence)
                st.markdown(f"• **{obj.name}** - <span style='color: {color};'>{confidence_pct:.1f}%</span>",
                            unsafe_allow_html=True)

        with col2:
            fig_objects = create_confidence_chart(result.objects, 'name', 'confidence')
            if fig_objects:
                st.plotly_chart(fig_objects, use_container_width=True)

    # OCR Text
    if result.has_text:
        st.markdown("### 📝 Extracted Text (OCR)")

        with st.expander("View Extracted Text", expanded=False):
            for i, ocr in enumerate(result.ocr_text, 1):
                confidence_pct = ocr.confidence * 100
                color = get_confidence_color(ocr.confidence)
                st.markdown(f"{i}. {ocr.text} - <span style='color: {color};'>{confidence_pct:.1f}%</span>",
                            unsafe_allow_html=True)

    # Faces
    if result.has_faces:
        st.markdown("### 👤 Faces Detected")
        for i, face in enumerate(result.faces, 1):
            age_info = f"Age: ~{face.age}" if face.age else "Age: Unknown"
            gender_info = f", Gender: {face.gender}" if face.gender else ""
            st.info(f"Face {i}: {age_info}{gender_info}")

    # Colors
    if result.dominant_colors:
        st.markdown("### 🎨 Color Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Dominant Colors:**")
            for color in result.dominant_colors:
                st.markdown(f"• {color}")

        with col2:
            if result.accent_color:
                st.markdown(f"**Accent Color:** {result.accent_color}")

        with col3:
            if result.is_bw_image:
                st.markdown("**Black & White:** Yes")

    # Suggestions
    if summary['suggestions']:
        st.markdown("### 💡 Suggestions for Better Results")
        for suggestion in summary['suggestions']:
            st.info(suggestion)

    # Download results
    st.markdown("### 💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        # JSON download
        json_data = json.dumps(result.to_dict(), indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name="analysis_results.json",
            mime="application/json"
        )

    with col2:
        # Text download
        text_data = f"""Image Analysis Results
{'=' * 60}

Descriptions:
{chr(10).join([f"- {d.text} ({d.confidence_percentage:.1f}%)" for d in result.descriptions])}

Tags:
{chr(10).join([f"- {t.name} ({t.confidence_percentage:.1f}%)" for t in result.tags[:10]])}

Objects:
{chr(10).join([f"- {o.name} ({o.confidence_percentage:.1f}%)" for o in result.objects])}

Reliability Score: {reliability['reliability_score'] * 100:.1f}%
"""
        st.download_button(
            label="📝 Download Text",
            data=text_data,
            file_name="analysis_results.txt",
            mime="text/plain"
        )


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<p class="main-header">📸 Image Analyzer Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Azure Computer Vision AI</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Confidence threshold
        min_confidence = st.slider(
            "Minimum Confidence Threshold",
            0.0, 1.0, 0.3,
            help="Filter results below this confidence level"
        )

        # Include OCR
        include_ocr = st.checkbox(
            "Include Text Extraction (OCR)",
            value=True,
            help="Extract text from image"
        )

        st.divider()

        st.header("ℹ️ About")
        st.info("""
        This app uses Azure Computer Vision to analyze images and provide:
        - Descriptions
        - Object detection
        - Tag generation
        - Text extraction (OCR)
        - Content moderation
        - Color analysis

        All results include confidence scores and uncertainty handling.
        """)

        # Analysis history
        if st.session_state.analysis_history:
            st.divider()
            st.header("📜 History")
            st.markdown(f"Analyzed {len(st.session_state.analysis_history)} images")

    # Main content
    tabs = st.tabs(["🔍 Analyze Image", "📊 Batch Analysis", "📖 Guide"])

    with tabs[0]:
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="Upload an image to analyze"
        )

        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 2])

            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Image info
                st.markdown(f"""
                **Image Info:**
                - Size: {image.width} x {image.height}
                - Format: {image.format}
                - Mode: {image.mode}
                """)

            with col2:
                # Analyze button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image... Please wait..."):
                        try:
                            # Convert to bytes
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format=image.format or 'PNG')
                            img_byte_arr = img_byte_arr.getvalue()

                            # Analyze
                            result = st.session_state.analyzer.analyze_from_bytes(
                                img_byte_arr,
                                include_ocr=include_ocr
                            )

                            # Store in session
                            st.session_state.current_result = result
                            st.session_state.current_image = image
                            st.session_state.analysis_history.append({
                                'filename': uploaded_file.name,
                                'timestamp': pd.Timestamp.now()
                            })

                            st.success("✅ Analysis complete!")

                        except AnalysisError as e:
                            st.error(f"❌ Analysis Error: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Unexpected Error: {str(e)}")

            # Display results
            if hasattr(st.session_state, 'current_result'):
                st.divider()
                display_analysis_results(
                    st.session_state.current_result,
                    st.session_state.current_image
                )

    with tabs[1]:
        st.markdown("### 📊 Batch Analysis")
        st.info("Upload multiple images for batch processing (Coming Soon!)")

    with tabs[2]:
        st.markdown("### 📖 User Guide")

        st.markdown("""
        #### How to Use

        1. **Upload an Image**
           - Click "Browse files" or drag and drop an image
           - Supported formats: JPG, PNG, BMP, GIF
           - Max size: 4MB

        2. **Configure Settings** (Sidebar)
           - Adjust confidence threshold
           - Enable/disable OCR text extraction

        3. **Analyze**
           - Click "Analyze Image" button
           - Wait for results

        4. **Review Results**
           - View descriptions, tags, and objects
           - Check confidence scores
           - Read uncertainty warnings
           - Download results as JSON or text

        #### Understanding Confidence Scores

        - 🟢 **High (80%+)**: Very reliable
        - 🟡 **Medium (50-80%)**: Likely correct, verify if critical
        - 🟠 **Low (30-50%)**: Use with caution
        - 🔴 **Very Low (<30%)**: Unreliable

        #### Tips for Better Results

        - Use high-resolution images
        - Ensure good lighting
        - Avoid blurry or distorted images
        - For OCR, ensure text is clearly readable
        """)


if __name__ == "__main__":
    main()