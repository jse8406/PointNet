import torch

# CUDA가 사용 가능한지 확인하는 출력
print(f"CUDA is available: {torch.cuda.is_available()}")

# 현재 사용 가능한 CUDA 버전 출력
print(f"CUDA version: {torch.version.cuda}")

# 현재 설치된 PyTorch 버전 출력
print(f"PyTorch version: {torch.__version__}")

# 사용 가능한 GPU의 수 출력
print(f"Number of GPUs: {torch.cuda.device_count()}")

# GPU가 사용 가능한 경우, GPU 이름을 출력
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#https://discuss.pytorch.org/t/torch-cuda-is-available-is-false-cuda-12-2-rtx-4070/185250 참고