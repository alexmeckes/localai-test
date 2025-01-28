from huggingface_hub import HfApi
from anthropic import Anthropic
import json
from typing import Optional, Dict, Tuple
from app.model_types import (
    ModelRequirements, ModelFramework, PrecisionType,
    QuantizationType, ModelArchitecture
)

class ModelInfoService:
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.hf_api = HfApi()
        self.anthropic = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        self._cache: Dict[str, ModelRequirements] = {}
    
    def get_model_card(self, model_id: str) -> str:
        """Fetch model card from HuggingFace."""
        try:
            model_info = self.hf_api.model_info(model_id)
            return model_info.card_data
        except Exception as e:
            return f"Error fetching model card: {str(e)}"
    
    def estimate_memory_requirements(
        self,
        param_count: float,
        precision: PrecisionType,
        quantization: Optional[QuantizationType],
        architecture: Optional[ModelArchitecture]
    ) -> Tuple[float, float, float, bool]:
        """
        Estimate memory requirements based on model characteristics.
        Returns: (min_gpu_memory, recommended_gpu_memory, min_cpu_ram, can_run_cpu)
        """
        # Base memory per parameter in bytes
        bytes_per_param = 4  # Default to FP32
        
        # Adjust based on precision
        if precision == PrecisionType.FP16:
            bytes_per_param = 2
        elif precision == PrecisionType.INT8:
            bytes_per_param = 1
        elif precision == PrecisionType.INT4:
            bytes_per_param = 0.5
        elif precision == PrecisionType.MIXED:
            bytes_per_param = 2  # Conservative estimate for mixed precision

        # Further adjust for quantization if specified
        if quantization:
            if quantization in [QuantizationType.GGUF, QuantizationType.GGML]:
                bytes_per_param = 0.5  # ~4-bit quantization
            elif quantization in [QuantizationType.GPTQ, QuantizationType.AWQ]:
                bytes_per_param = 0.4
            elif quantization == QuantizationType.BITSANDBYTES:
                bytes_per_param = 0.5

        # Base memory calculation in GB
        base_memory = (param_count * bytes_per_param * 1e9) / (1024**3)  # Convert bytes to GB

        # Architecture-specific overhead
        if architecture == ModelArchitecture.DECODER_ONLY:
            # Add memory for KV cache - scales with model size but with diminishing returns
            kv_cache = min(base_memory * 0.05, 4)  # Cap at 4GB for KV cache
            base_memory += kv_cache
        elif architecture == ModelArchitecture.ENCODER_DECODER:
            # Add memory for cross-attention
            base_memory *= 1.1

        # Scale overhead factors based on model size
        # Smaller models need relatively more overhead
        if param_count < 3:  # Under 3B parameters
            cuda_overhead = 1.1
            safety_margin = 1.3
            cpu_overhead = 1.2
        elif param_count < 7:  # 3B-7B parameters
            cuda_overhead = 1.15
            safety_margin = 1.4
            cpu_overhead = 1.3
        else:  # Over 7B parameters
            cuda_overhead = 1.2
            safety_margin = 1.5
            cpu_overhead = 1.5

        # Minimum GPU memory (add overhead for CUDA tensors and gradients)
        min_gpu_memory = base_memory * cuda_overhead

        # Recommended GPU memory (add safety margin)
        recommended_gpu_memory = min_gpu_memory * safety_margin

        # CPU RAM requirements (need more memory for CPU inference)
        min_cpu_ram = base_memory * cpu_overhead + 2  # Add 2GB baseline for processing

        # CPU compatibility
        can_run_cpu = False
        if quantization in [QuantizationType.GGUF, QuantizationType.GGML]:
            can_run_cpu = True
        elif param_count < 13:  # 13B parameters threshold
            can_run_cpu = True
        
        return min_gpu_memory, recommended_gpu_memory, min_cpu_ram, can_run_cpu

    def extract_model_requirements(self, model_id: str) -> ModelRequirements:
        """Extract model requirements using Claude."""
        if model_id in self._cache:
            return self._cache[model_id]
        
        model_card = self.get_model_card(model_id)
        
        if not self.anthropic:
            raise ValueError("Anthropic API key not provided")
        
        prompt = f"""Based on this model card, extract the model requirements and return them in this exact JSON format without any additional text or markdown:
        {{
            "parameter_count": {{"value": null, "confidence": 0.0, "quote": ""}},
            "framework": {{"value": "pytorch", "confidence": 0.0, "quote": ""}},
            "precision": {{"value": "fp32", "confidence": 0.0, "quote": ""}},
            "quantization": {{"value": null, "confidence": 0.0, "quote": ""}},
            "architecture": {{"value": null, "confidence": 0.0, "quote": ""}},
            "memory_requirements": {{"value": 0, "confidence": 0.0, "quote": ""}},
            "min_gpu_memory": {{"value": 0, "confidence": 0.0, "quote": ""}},
            "recommended_gpu_memory": {{"value": 0, "confidence": 0.0, "quote": ""}},
            "can_run_cpu": {{"value": false, "confidence": 0.0, "quote": ""}},
            "min_cpu_ram": {{"value": 0, "confidence": 0.0, "quote": ""}}
        }}

        Fill in the values based on the model card information. For each field:
        - Set appropriate numeric values (parameter_count in billions)
        - Set confidence score (0-1) based on how certain you are
        - Include relevant quote from the text that supports your value
        - For framework, use one of: pytorch, tensorflow, jax, onnx, other
        - For precision, use one of: fp16, fp32, int8, int4, mixed
        - For quantization, use one of: none, gguf, ggml, gptq, awq, qlora, bitsandbytes
        - For architecture, use one of: decoder_only, encoder_decoder, encoder_only, other
          Look for these indicators:
          - decoder_only: LLMs, GPT-style models, text generation models
          - encoder_decoder: T5-style, BART-style, sequence-to-sequence models
          - encoder_only: BERT-style, RoBERTa-style, embedding models
          - other: if architecture is unclear or different

        Look for:
        1. Explicit memory requirements in the model card
        2. Quantization method used (GGUF, GPTQ, AWQ, etc.)
        3. Architecture type indicators:
           - Model family/base architecture (e.g., T5, BERT, GPT)
           - Task type (text generation, translation, embeddings)
           - Model structure descriptions
        4. Precision format (FP16, INT8, etc.)
        5. CPU compatibility information

        Set architecture confidence lower (0.7 or less) if inferred only from task type or model family.
        Set confidence to 0.9+ only if architecture is explicitly stated.

        Model Card:
        {model_card}
        """
        
        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0,
            system="You are an AI trained to extract model requirements from model cards. Output only valid JSON without any additional text or explanation.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            # Parse the response content
            content = response.content[0].text if isinstance(response.content, list) else response.content
            
            # Try to find JSON in the response if it's wrapped in markdown
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            # Calculate overall confidence
            confidences = [v.get('confidence', 0) for v in result.values()]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Create source quotes dictionary
            source_quotes = {k: v.get('quote', '') for k, v in result.items()}
            
            # Handle null values with defaults
            framework_value = result['framework']['value']
            if framework_value is None or framework_value not in [e.value for e in ModelFramework]:
                framework_value = 'other'
                
            precision_value = result['precision']['value']
            if precision_value is None or precision_value not in [e.value for e in PrecisionType]:
                precision_value = 'fp32'

            # Extract or estimate memory requirements
            param_count = result['parameter_count']['value'] or 0
            quantization_value = result['quantization']['value']
            if quantization_value and quantization_value not in [e.value for e in QuantizationType]:
                quantization_value = None
            
            architecture_value = result['architecture']['value']
            if architecture_value and architecture_value not in [e.value for e in ModelArchitecture]:
                architecture_value = None

            # If memory requirements aren't explicitly stated, estimate them
            if not any([
                result['min_gpu_memory']['value'],
                result['recommended_gpu_memory']['value'],
                result['min_cpu_ram']['value']
            ]):
                min_gpu, rec_gpu, min_cpu, can_cpu = self.estimate_memory_requirements(
                    param_count,
                    PrecisionType(precision_value),
                    QuantizationType(quantization_value) if quantization_value else None,
                    ModelArchitecture(architecture_value) if architecture_value else None
                )
                
                # Only use estimated values if not provided in model card
                if not result['min_gpu_memory']['value']:
                    result['min_gpu_memory']['value'] = min_gpu
                if not result['recommended_gpu_memory']['value']:
                    result['recommended_gpu_memory']['value'] = rec_gpu
                if not result['min_cpu_ram']['value']:
                    result['min_cpu_ram']['value'] = min_cpu
                if not result['can_run_cpu']['value']:
                    result['can_run_cpu']['value'] = can_cpu
            
            requirements = ModelRequirements(
                name=model_id,
                parameter_count=param_count,
                memory_requirements=result['memory_requirements']['value'] or 0,
                framework=ModelFramework(framework_value),
                precision=PrecisionType(precision_value),
                quantization=QuantizationType(quantization_value) if quantization_value else None,
                architecture=ModelArchitecture(architecture_value) if architecture_value else None,
                min_gpu_memory=result['min_gpu_memory']['value'] or 0,
                recommended_gpu_memory=result['recommended_gpu_memory']['value'] or 0,
                can_run_cpu=result['can_run_cpu']['value'] or False,
                min_cpu_ram=result['min_cpu_ram']['value'] or 0,
                extraction_confidence=avg_confidence,
                source_quotes=source_quotes
            )
            
            self._cache[model_id] = requirements
            return requirements
            
        except Exception as e:
            raise ValueError(f"Failed to parse model requirements: {str(e)}")
    
    def clear_cache(self):
        """Clear the model requirements cache."""
        self._cache.clear() 