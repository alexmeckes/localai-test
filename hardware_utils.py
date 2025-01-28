import psutil
import platform
import os
from typing import Optional, List
import subprocess
from app.model_types import SystemSpecs, GPUInfo
import wmi
import GPUtil

def get_gpu_info() -> List[GPUInfo]:
    """Get GPU information using GPUtil."""
    gpus = []
    try:
        nvidia_gpus = GPUtil.getGPUs()
        for gpu in nvidia_gpus:
            gpus.append(GPUInfo(
                name=gpu.name,
                memory=gpu.memoryTotal / 1024  # Convert MB to GB
            ))
    except Exception as e:
        print(f"GPUtil Error: {str(e)}")
    return gpus

def get_cpu_info() -> tuple[str, int]:
    """Get CPU name and core count."""
    if platform.system() == "Windows":
        try:
            cpu_name = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split("\n")[1].strip()
        except Exception:
            cpu_name = platform.processor()
    else:
        cpu_name = platform.processor()
    
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    return cpu_name, cpu_cores

def get_system_specs() -> SystemSpecs:
    """Get complete system specifications."""
    memory = psutil.virtual_memory()
    total_ram = memory.total / (1024**3)  # Convert to GB
    available_ram = memory.available / (1024**3)  # Convert to GB
    
    gpus = get_gpu_info()
    cpu_name, cpu_cores = get_cpu_info()
    
    return SystemSpecs(
        total_ram=total_ram,
        available_ram=available_ram,
        gpus=gpus,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores
    ) 
