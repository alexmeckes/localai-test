from huggingface_hub import HfApi
from openai import OpenAI
import json
from typing import Optional, Dict, Tuple
from app.model_types import (
    ModelRequirements, ModelFramework, PrecisionType,
    QuantizationType, ModelArchitecture
)

class ModelInfoService:
    def __init__(self, openrouter_api_key: Optional[str] = None):
        self.hf_api = HfApi()
        self.openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo/ai-model-compatibility-checker",
                "X-Title": "AI Model Compatibility Checker"
            }
        ) if openrouter_api_key else None
        self._cache: Dict[str, ModelRequirements] = {}
    
    def get_model_card(self, model_id: str) -> str:
        """Fetch model card from HuggingFace."""
        try:
            # Remove any URL parts and get just the model ID
            if 'huggingface.co/' in model_id:
                model_id = model_id.split('huggingface.co/')[-1]
            print(f"Fetching model card for cleaned ID: {model_id}")  # Debug log
            
            model_info = self.hf_api.model_info(model_id)
            
            # Combine all available information
            model_text = []
            
            # Add model name and ID
            model_text.append(f"Model ID: {model_id}")
            if hasattr(model_info, 'modelId'):
                model_text.append(f"Model Name: {model_info.modelId}")
            
            # Add description if available
            if hasattr(model_info, 'description'):
                model_text.append(f"Description: {model_info.description}")
            
            # Add tags and other metadata
            if hasattr(model_info, 'tags'):
                model_text.append(f"Tags: {', '.join(model_info.tags)}")
            if hasattr(model_info, 'pipeline_tag'):
                model_text.append(f"Pipeline: {model_info.pipeline_tag}")
            if hasattr(model_info, 'library_name'):
                model_text.append(f"Framework: {model_info.library_name}")
            
            # Add model card data if available
            if hasattr(model_info, 'cardData'):
                model_text.append(str(model_info.cardData))
            elif hasattr(model_info, 'card_data'):
                model_text.append(str(model_info.card_data))
            
            # Add siblings (model files) information
            if hasattr(model_info, 'siblings'):
                files = [s.rfilename for s in model_info.siblings]
                model_text.append(f"Model files: {', '.join(files)}")
            
            combined_text = "\n\n".join(model_text)
            print(f"Retrieved model information length: {len(combined_text)} characters")  # Debug log
            print(f"Model information preview:\n{combined_text[:500]}...")  # Debug log
            return combined_text
            
        except Exception as e:
            print(f"Error details: {str(e)}")  # Debug log
            error_text = f"Error fetching model card: {str(e)}"
            print(f"Returning error text: {error_text}")  # Debug log
            return error_text
    
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

    def extract_parameters_from_text(self, text: str, model_id: str) -> Tuple[Optional[float], float, str]:
        """
        Extract parameter count from text using regex patterns.
        Returns: (parameter_count, confidence, quote)
        """
        import re
        
        # Common patterns for parameter counts
        patterns = [
            # Explicit patterns
            (r'(\d+\.?\d*)\s*B\s*parameters', 1.0),  # "7B parameters"
            (r'(\d+\.?\d*)\s*billion\s*parameters', 1.0),  # "7 billion parameters"
            (r'(\d+\.?\d*)\s*B\s*params', 1.0),  # "7B params"
            (r'(\d+\.?\d*)\s*billion\s*params', 1.0),  # "7 billion params"
            
            # Name-based patterns
            (r'[-_/](\d+\.?\d*)b[-_]', 0.8),  # "-7b-" in name
            (r'[-_/](\d+\.?\d*)B[-_]', 0.8),  # "-7B-" in name
            
            # Million-based patterns
            (r'(\d+\.?\d*)\s*M\s*parameters', 0.9),  # "350M parameters"
            (r'(\d+\.?\d*)\s*million\s*parameters', 0.9),  # "350 million parameters"
            (r'[-_/](\d+\.?\d*)m[-_]', 0.7),  # "-350m-" in name
            (r'[-_/](\d+\.?\d*)M[-_]', 0.7),  # "-350M-" in name
        ]
        
        for pattern, confidence in patterns:
            # Try in both the text and model_id
            for source in [text, model_id]:
                match = re.search(pattern, source, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # Convert millions to billions if needed
                    if 'M' in pattern or 'm' in pattern:
                        value = value / 1000
                    return value, confidence, match.group(0)
        
        return None, 0.0, ""

    def extract_model_requirements(self, model_id: str) -> ModelRequirements:
        """Extract model requirements using DeepSeek."""
        if model_id in self._cache:
            return self._cache[model_id]
        
        try:
            print(f"Fetching model card for {model_id}...")  # Debug log
            model_card = self.get_model_card(model_id)
            print(f"Model card content preview: {model_card[:200] if model_card else 'Empty card'}...")  # Debug log
            
            # Try regex-based parameter extraction first
            param_count, param_confidence, param_quote = self.extract_parameters_from_text(model_card, model_id)
            print(f"Regex parameter extraction: count={param_count}, confidence={param_confidence}, quote='{param_quote}'")
            
            if not self.openai:
                raise ValueError("OpenRouter API key not provided")
            
            system_prompt = """You are an expert AI system designed to analyze machine learning model specifications.
Your task is to carefully analyze model documentation and extract precise technical requirements.
Follow these principles:
1. Be thorough in examining all technical details
2. Use logical reasoning to infer requirements when not explicitly stated
3. Maintain high precision in numerical values
4. Set confidence scores based on evidence quality
5. Always include supporting quotes for your conclusions
6. Return only valid JSON without any additional text

For parameter count extraction:
- Look for patterns like "7B", "7b", "7 billion", "7 B parameters"
- Check model names for size indicators (e.g., "llama-7b" means 7 billion parameters)
- Convert millions to billions (e.g., "350M" = 0.35B)
- Set high confidence (0.9+) for explicit mentions
- Set medium confidence (0.5-0.8) for name-based inference
- Include the exact quote that supports your conclusion

Think step by step:
1. First scan for explicit parameter counts
2. If not found, check model name for size indicators
3. If still not found, look for indirect references
4. Validate and convert to billions
5. Set confidence based on source quality"""

            user_prompt = f"""Analyze this model card and extract the technical requirements. Return the information in this exact JSON format:

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

            Requirements:
            1. Parameter count: Extract the model size in billions (e.g., 7 for a 7B model)
               - Look for explicit mentions: "7B parameters", "7 billion", etc.
               - Check model name for size indicators: "llama-7b", "opt-350m", etc.
               - Convert millions to billions: 350M = 0.35B
               - Common patterns:
                 * "7B" or "7b" → 7 billion
                 * "350M" or "350m" → 0.35 billion
                 * "7 billion parameters" → 7 billion
                 * "7 B params" → 7 billion
               - Set confidence based on source:
                 * Explicit mention in text: 0.9-1.0
                 * Size in model name: 0.7-0.8
                 * Indirect reference: 0.5-0.6

            2. Framework: Identify the deep learning framework
               - Valid options: pytorch, tensorflow, jax, onnx, other
               - Look for framework-specific terms or file formats

            3. Precision: Determine numerical precision format
               - Valid options: fp16, fp32, int8, int4, mixed
               - Check for terms like "half precision", "mixed precision"

            4. Quantization: Identify any quantization method
               - Valid options: none, gguf, ggml, gptq, awq, qlora, bitsandbytes
               - Look for terms about model compression or optimization

            5. Architecture: Determine model architecture type
               - Valid options: decoder_only, encoder_decoder, encoder_only, other
               - Analyze model family and task type:
                 * decoder_only: LLMs, GPT-style models
                 * encoder_decoder: T5, BART, translation models
                 * encoder_only: BERT, RoBERTa, embedding models

            6. Memory Requirements: Extract or infer memory needs
               - Look for explicit GPU/RAM requirements
               - Consider model size and architecture type
               - Note any CPU compatibility information

            For each field:
            - Set confidence score (0.0-1.0) based on evidence quality
            - Include relevant quotes that support your conclusion
            - Use null for unknown values rather than guessing
            - Prefer explicit information over inferred

            Model Card:
            {model_card}
            """
            
            # Prepare a shorter version of the model card if it's too long
            max_card_length = 8000  # Maximum length for the model card
            if len(model_card) > max_card_length:
                print(f"Model card too long ({len(model_card)} chars), truncating...")
                model_card = model_card[:max_card_length] + "..."
            
            print("Making OpenRouter API call...")  # Debug log
            try:
                response = self.openai.chat.completions.create(
                    model="deepseek/deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    top_p=0.9,
                    timeout=30  # Add timeout
                )
                print("OpenRouter API call completed successfully")  # Debug log
            except Exception as api_error:
                print(f"OpenRouter API call failed: {str(api_error)}")
                # If API call fails, create a default result with regex-extracted parameters
                result = {
                    "parameter_count": {"value": param_count, "confidence": param_confidence, "quote": param_quote},
                    "framework": {"value": "pytorch", "confidence": 0.5, "quote": "Default framework"},
                    "precision": {"value": "fp32", "confidence": 0.5, "quote": "Default precision"},
                    "quantization": {"value": None, "confidence": 0.0, "quote": ""},
                    "architecture": {"value": None, "confidence": 0.0, "quote": ""},
                    "memory_requirements": {"value": 0, "confidence": 0.0, "quote": ""},
                    "min_gpu_memory": {"value": 0, "confidence": 0.0, "quote": ""},
                    "recommended_gpu_memory": {"value": 0, "confidence": 0.0, "quote": ""},
                    "can_run_cpu": {"value": True, "confidence": 0.5, "quote": "Default CPU compatibility"},
                    "min_cpu_ram": {"value": 0, "confidence": 0.0, "quote": ""}
                }
            else:
                # Parse the response content
                content = response.choices[0].message.content
                print(f"Full response content: {content}")  # Debug log full content
                
                # Try to find JSON in the response if it's wrapped in markdown
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                try:
                    result = json.loads(content)
                    print(f"Parsed JSON result: {json.dumps(result, indent=2)}")  # Debug log parsed JSON
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    print(f"Content that failed to parse: {content}")
                    # Use regex results if JSON parsing fails
                    result = {
                        "parameter_count": {"value": param_count, "confidence": param_confidence, "quote": param_quote},
                        "framework": {"value": "pytorch", "confidence": 0.5, "quote": "Default framework"},
                        "precision": {"value": "fp32", "confidence": 0.5, "quote": "Default precision"},
                        "quantization": {"value": None, "confidence": 0.0, "quote": ""},
                        "architecture": {"value": None, "confidence": 0.0, "quote": ""},
                        "memory_requirements": {"value": 0, "confidence": 0.0, "quote": ""},
                        "min_gpu_memory": {"value": 0, "confidence": 0.0, "quote": ""},
                        "recommended_gpu_memory": {"value": 0, "confidence": 0.0, "quote": ""},
                        "can_run_cpu": {"value": True, "confidence": 0.5, "quote": "Default CPU compatibility"},
                        "min_cpu_ram": {"value": 0, "confidence": 0.0, "quote": ""}
                    }
            
            # Calculate overall confidence
            confidences = [v.get('confidence', 0) for v in result.values()]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average confidence: {avg_confidence}")  # Debug log confidence
            
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
            
            # If LLM didn't find parameters but regex did, use regex results
            if not result['parameter_count']['value'] and param_count is not None:
                result['parameter_count']['value'] = param_count
                result['parameter_count']['confidence'] = param_confidence
                result['parameter_count']['quote'] = param_quote
                print(f"Using regex-extracted parameters: {param_count}B")
            
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
            
            print(f"Final requirements object: {requirements}")  # Debug log final object
            self._cache[model_id] = requirements
            return requirements
            
        except Exception as e:
            print(f"Error in extract_model_requirements: {str(e)}")  # Debug log error
            import traceback
            print(f"Traceback: {traceback.format_exc()}")  # Debug log traceback
            raise
    
    def clear_cache(self):
        """Clear the model requirements cache."""
        self._cache.clear() 