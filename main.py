import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from hardware_utils import get_system_specs
from model_service import ModelInfoService
from model_types import PrecisionType, QuantizationType
import plotly.graph_objects as go

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# ... existing code ... 