import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from hardware_utils import get_system_specs
from model_service import ModelInfoService
from model_types import PrecisionType, QuantizationType
import plotly.graph_objects as go

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Page config
st.set_page_config(
    page_title="AI Model Compatibility Checker",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stAlert {
        border-radius: 5px;
    }
    .main > div {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stSelectbox label {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def init_model_service():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.warning("⚠️ ANTHROPIC_API_KEY not found in environment variables. Some features will be limited.")
    return ModelInfoService(api_key)

model_service = init_model_service()

# Get system specs
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_system_specs():
    return get_system_specs()

# UI Components
def render_header():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🤖 AI Model Compatibility Checker")
        st.markdown("""
        Check if your hardware can run specific AI models locally. Enter a HuggingFace model ID and configure inference settings below.
        """)
    with col2:
        st.markdown("""
        ### Quick Tips
        - 💡 Lower precision/quantization = less memory needed
        - 💾 4-bit quantization can reduce memory by ~8x
        - 🔄 Memory estimates update automatically
        """)

def render_system_specs(specs):
    with st.expander("💻 Your System Specifications", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CPU & Memory")
            st.metric("CPU", specs.cpu_name, help="Your CPU model")
            st.metric("CPU Cores", specs.cpu_cores, help="Number of physical CPU cores")
            st.metric("Total RAM", f"{specs.total_ram:.1f} GB", help="Total system memory")
        
        with col2:
            st.markdown("#### GPU Information")
            if specs.gpus:
                for i, gpu in enumerate(specs.gpus, 1):
                    st.metric(f"GPU {i}", gpu.name, help=f"GPU {i} model")
                    st.metric(f"GPU {i} Memory", f"{gpu.memory:.1f} GB", help=f"Available VRAM on GPU {i}")
            else:
                st.info("No GPU detected", icon="ℹ️")

def render_model_selection():
    st.markdown("### 🔍 Model Selection")
    model_id = st.text_input(
        "Enter HuggingFace Model ID",
        placeholder="e.g., facebook/opt-350m",
        help="Enter the model ID from HuggingFace Hub (e.g., facebook/opt-350m)"
    )
    
    st.markdown("### ⚙️ Inference Configuration")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; font-size: 0.9em;">
        ℹ️ Since model cards often don't specify all available inference configurations, please manually select the options you know are available for your use case.
    </div>
    """, unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        selected_precision = st.selectbox(
            "Precision",
            options=[p.value for p in PrecisionType],
            help="Lower precision = less memory but potentially lower quality\n\n" +
                 "- FP32: Full precision, highest accuracy\n" +
                 "- FP16: Half precision, good balance\n" +
                 "- INT8/INT4: Reduced precision, significant memory savings"
        )
    
    with config_col2:
        selected_quantization = st.selectbox(
            "Quantization",
            options=["none"] + [q.value for q in QuantizationType],
            help="Quantization reduces model size and memory usage\n\n" +
                 "- GGUF/GGML: Optimized for CPU inference\n" +
                 "- GPTQ/AWQ: Optimized for GPU inference\n" +
                 "- QLORA: For fine-tuning with limited memory"
        )
    
    return model_id, selected_precision, selected_quantization

def render_compatibility_gauge(score):
    colors = {
        'poor': '#ff4b4b',
        'fair': '#faa356',
        'good': '#2ea043'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': colors['good'] if score > 0.7 else colors['fair'] if score > 0.3 else colors['poor']},
            'steps': [
                {'range': [0, 33], 'color': 'rgba(255, 75, 75, 0.2)'},
                {'range': [33, 66], 'color': 'rgba(250, 163, 86, 0.2)'},
                {'range': [66, 100], 'color': 'rgba(46, 160, 67, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_model_requirements(model_reqs):
    st.markdown("### 📊 Model Requirements")
    
    # Basic info
    basic_col1, basic_col2, basic_col3 = st.columns(3)
    with basic_col1:
        st.metric("Parameters", f"{model_reqs.parameter_count:.1f}B" if model_reqs.parameter_count else "Unknown",
                 help="Model size in billions of parameters")
        st.metric("Framework", model_reqs.framework.value,
                 help="Deep learning framework used by the model")
    
    with basic_col2:
        st.metric("Architecture", 
                 model_reqs.architecture.value if model_reqs.architecture else "Unknown",
                 help="Model architecture type (e.g., decoder-only for LLMs)")
        st.metric("Can Run on CPU", "Yes ✅" if model_reqs.can_run_cpu else "No ❌",
                 help="Whether the model can run on CPU-only systems")
    
    with basic_col3:
        st.metric("Extraction Confidence", f"{model_reqs.extraction_confidence:.0%}",
                 help="Confidence in the extracted model requirements")
    
    # Memory requirements
    st.markdown("#### 💾 Memory Requirements")
    
    with st.expander("ℹ️ How are these requirements estimated?"):
        st.markdown("""
        1. **Base Memory (per parameter)**
           - FP32: 4 bytes
           - FP16: 2 bytes
           - INT8: 1 byte
           - INT4: 0.5 bytes
           - Mixed: ~2 bytes (varies)
        
        2. **Quantization Effects**
           - GGUF/GGML: ~0.5 bytes per parameter
           - GPTQ/AWQ: ~0.4 bytes per parameter
           - QLORA: varies based on configuration
        
        3. **Additional Overhead**
           - Small models (<3B params): +10% CUDA overhead, +30% safety margin
           - Medium models (3-7B): +15% CUDA overhead, +40% safety margin
           - Large models (>7B): +20% CUDA overhead, +50% safety margin
           
        4. **Architecture-Specific**
           - Decoder-only: Additional memory for KV cache
           - Encoder-decoder: +10% for cross-attention
        
        5. **CPU Requirements**
           - Base memory + 20-50% overhead
           - +2GB baseline for processing
           
        > Note: These are estimates unless explicitly stated in the model card.
        > Actual requirements may vary based on implementation and usage.
        """)
    
    mem_col1, mem_col2 = st.columns(2)
    
    with mem_col1:
        st.metric(
            "GPU Memory (Minimum)", 
            f"{model_reqs.min_gpu_memory:.1f} GB",
            help="Minimum GPU memory required for inference\n\n" +
                 ("🔄 Estimated based on model characteristics" if not model_reqs.source_quotes.get('min_gpu_memory') else "📝 Value from model card")
        )
        st.metric(
            "GPU Memory (Recommended)", 
            f"{model_reqs.recommended_gpu_memory:.1f} GB",
            help="Recommended GPU memory for optimal performance\n\n" +
                 ("🔄 Estimated: minimum + safety margin" if not model_reqs.source_quotes.get('recommended_gpu_memory') else "📝 Value from model card")
        )
    
    with mem_col2:
        st.metric(
            "CPU RAM Required", 
            f"{model_reqs.min_cpu_ram:.1f} GB",
            help="Minimum system RAM required for CPU inference\n\n" +
                 ("🔄 Estimated based on model characteristics" if not model_reqs.source_quotes.get('min_cpu_ram') else "📝 Value from model card")
        )

def render_compatibility_messages(messages):
    st.markdown("### 🎯 Compatibility Analysis")
    
    # Separate messages by type
    status_messages = []
    info_messages = []
    detail_messages = []
    
    for msg in messages:
        if msg.startswith("✅") or msg.startswith("❌") or msg.startswith("⚠️"):
            status_messages.append(msg)
        elif msg.startswith("ℹ️"):
            info_messages.append(msg)
        else:
            detail_messages.append(msg)
    
    # Display messages by type
    for msg in status_messages:
        st.markdown(msg)
    
    if info_messages:
        with st.expander("Additional GPU Information", expanded=True):
            for msg in info_messages:
                st.markdown(msg)
    
    if detail_messages:
        with st.expander("Requirement Details", expanded=True):
            for msg in detail_messages:
                st.markdown(msg)

def check_compatibility(system_specs, model_reqs):
    scores = []
    messages = []
    
    # Check GPU compatibility
    if system_specs.gpus:
        # Check all available GPUs
        compatible_gpus = []
        for gpu in system_specs.gpus:
            if gpu.memory >= model_reqs.recommended_gpu_memory:
                compatible_gpus.append((gpu, "recommended"))
            elif gpu.memory >= model_reqs.min_gpu_memory:
                compatible_gpus.append((gpu, "minimum"))
        
        if compatible_gpus:
            best_match = compatible_gpus[0]  # Most compatible GPU
            if best_match[1] == "recommended":
                scores.append(1.0)
                messages.append(f"✅ GPU ({best_match[0].name}) has sufficient memory ({best_match[0].memory:.1f} GB)")
            else:
                scores.append(0.7)
                messages.append(f"⚠️ GPU ({best_match[0].name}) meets minimum but not recommended requirements")
            
            # Show info about other GPUs
            for gpu, _ in compatible_gpus[1:]:
                messages.append(f"ℹ️ Additional GPU available: {gpu.name} ({gpu.memory:.1f} GB)")
        else:
            scores.append(0.0)
            messages.append("❌ No GPU with sufficient memory found:")
            for gpu in system_specs.gpus:
                messages.append(f"   • {gpu.name}: {gpu.memory:.1f} GB available")
    elif not model_reqs.can_run_cpu:
        scores.append(0.0)
        messages.append("❌ Model requires GPU but none detected")
    
    # Check CPU RAM
    if system_specs.total_ram >= model_reqs.min_cpu_ram:
        scores.append(1.0)
        messages.append(f"✅ Total RAM ({system_specs.total_ram:.1f} GB) meets requirements")
    else:
        scores.append(0.0)
        messages.append(f"❌ Insufficient RAM ({system_specs.total_ram:.1f} GB total, {model_reqs.min_cpu_ram:.1f} GB required)")
    
    # Calculate overall score
    overall_score = sum(scores) / len(scores)
    
    # Add requirement source information
    messages.append("\n📊 Requirement Details:")
    if model_reqs.source_quotes.get('min_gpu_memory', '').strip():
        messages.append(f"GPU Memory (from model card): {model_reqs.source_quotes['min_gpu_memory']}")
    else:
        messages.append("GPU Memory requirements are estimated based on model size")
        
    if model_reqs.source_quotes.get('min_cpu_ram', '').strip():
        messages.append(f"RAM Requirements (from model card): {model_reqs.source_quotes['min_cpu_ram']}")
    else:
        messages.append("RAM requirements are estimated based on model size")
    
    return overall_score, messages

def main():
    render_header()
    
    # Get and display system specs
    system_specs = get_cached_system_specs()
    render_system_specs(system_specs)
    
    # Model selection and configuration
    model_id, selected_precision, selected_quantization = render_model_selection()
    
    if model_id:
        try:
            with st.spinner("Analyzing model requirements..."):
                model_reqs = model_service.extract_model_requirements(model_id)
                
                # Override precision and quantization with user selections
                model_reqs.precision = PrecisionType(selected_precision)
                model_reqs.quantization = QuantizationType(selected_quantization) if selected_quantization != "none" else None
                
                # Recalculate memory requirements with new settings
                min_gpu, rec_gpu, min_cpu, can_cpu = model_service.estimate_memory_requirements(
                    model_reqs.parameter_count,
                    model_reqs.precision,
                    model_reqs.quantization,
                    model_reqs.architecture
                )
                
                model_reqs.min_gpu_memory = min_gpu
                model_reqs.recommended_gpu_memory = rec_gpu
                model_reqs.min_cpu_ram = min_cpu
                model_reqs.can_run_cpu = can_cpu
            
            # Display results
            render_model_requirements(model_reqs)
            
            score, messages = check_compatibility(system_specs, model_reqs)
            render_compatibility_gauge(score)
            render_compatibility_messages(messages)
            
            # Source quotes in expander
            if model_reqs.source_quotes:
                with st.expander("🔍 View Source Information"):
                    for key, quote in model_reqs.source_quotes.items():
                        if quote:
                            st.markdown(f"**{key}**: {quote}")
            
        except Exception as e:
            st.error(f"Error analyzing model: {str(e)}")

if __name__ == "__main__":
    main() 