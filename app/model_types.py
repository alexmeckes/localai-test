from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    OTHER = "other"

class PrecisionType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"

class QuantizationType(str, Enum):
    NONE = "none"
    GGUF = "gguf"
    GGML = "ggml"
    GPTQ = "gptq"
    AWQ = "awq"
    QLORA = "qlora"
    BITSANDBYTES = "bitsandbytes"

class ModelArchitecture(str, Enum):
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
    ENCODER_ONLY = "encoder_only"
    OTHER = "other"

class GPUInfo(BaseModel):
    name: str
    memory: float  # in GB

class ModelRequirements(BaseModel):
    name: str
    parameter_count: Optional[float] = None
    memory_requirements: float  # in GB
    framework: ModelFramework
    precision: PrecisionType
    quantization: Optional[QuantizationType] = None
    architecture: Optional[ModelArchitecture] = None
    min_gpu_memory: float  # in GB
    recommended_gpu_memory: float  # in GB
    can_run_cpu: bool = False
    min_cpu_ram: float  # in GB
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    source_quotes: Optional[Dict[str, str]] = None
    
class SystemSpecs(BaseModel):
    total_ram: float  # in GB
    available_ram: float  # in GB
    gpus: List[GPUInfo] = []
    cpu_name: str
    cpu_cores: int 