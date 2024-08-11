import torch

from torch.utils.data import DataLoader
from data_preprocess import *
from model import *
from eval_model import *
from data_augment import *
# cuda check
if __name__ == "__main__":
    # cuda check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 데이터 로드
    start_idx = 1000
    base_path = 'E:\set2'
    # 컴퓨팅 자원 관계상 idx를 바꿔가며 1000개의 파일씩 학습을 진행
    
    train_data = load_data_from_directory(base_path, 'training',start_idx) # ('car', PointCloud with 1323 points.)
    val_data = load_data_from_directory(base_path, 'validation', start_idx)
    test_data = load_data_from_directory(base_path, 'test', start_idx)

    print(f'Train data: {len(train_data)}')
    print(count_classes(train_data))

    # 데이터셋 생성
    train_dataset = PointCloudDataset(train_data)
    val_dataset = PointCloudDataset(val_data)
    test_dataset = PointCloudDataset(test_data)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 생성
    pointnet = PointNet()
    pointnet.to(device)

    # 모델 로드
    model_path = 'pointnet_model.pth'
    if os.path.exists(model_path):
        pointnet.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'Model loaded from {model_path}')
    else:
        print(f'No model found at {model_path}, starting from scratch.')
        
    # 학습
    train_model(pointnet, train_loader, val_loader)
    # 테스트 데이터셋으로 평가
    num_classes = 9 # 0~8 0은 배경 클래스라고 가정
    # test_model(pointnet, test_loader, num_classes)
