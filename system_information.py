import platform
import psutil
import socket
import os
import subprocess
import json

def get_system_info():
    return {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "OS Release": platform.release(),
        "Architecture": platform.architecture()[0],
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "RAM Total (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }

def get_network_info():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        interfaces = psutil.net_if_addrs()
        return {
            "Hostname": hostname,
            "IP Address": ip_address,
            "Interfaces": {iface: [addr.address for addr in addrs if addr.family == socket.AF_INET] for iface, addrs in interfaces.items()}
        }
    except Exception as e:
        return {"error": f"Failed to get network info: {str(e)}"}

def get_gpu_info():
    try:
        if os.name == "nt":  # Windows
            output = subprocess.check_output("wmic path win32_videocontroller get name", shell=True).decode()
            gpus = [line.strip() for line in output.split("\n") if line.strip()][1:]
        else:  # Linux & Mac
            output = subprocess.check_output("lspci | grep -i nvidia", shell=True).decode()
            gpus = [line.split(': ')[-1] for line in output.split("\n") if line.strip()]
        return {"GPU(s)": gpus if gpus else ["No dedicated GPU detected"]}
    except Exception:
        return {"GPU(s)": ["GPU detection failed or no GPU present"]}

def get_software_info():
    return {
        "Python Version": platform.python_version(),
        "Installed Packages": list(pkg.split('==')[0] for pkg in subprocess.check_output(["pip", "list", "--format=freeze"]).decode().split("\n") if pkg)
    }

def main():
    system_info = get_system_info()
    network_info = get_network_info()
    gpu_info = get_gpu_info()
    software_info = get_software_info()
    
    info = {
        "System Information": system_info,
        "Network Information": network_info,
        "GPU Information": gpu_info,
        "Software Information": software_info
    }
    
    print(json.dumps(info, indent=4))
    
if __name__ == "__main__":
    main()
