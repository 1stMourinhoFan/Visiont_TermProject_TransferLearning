import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import defaultdict
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """모델 학습 함수 (scheduler 추가)"""
    train_losses = []
    val_losses = []

    # CUDA 이벤트를 사용한 GPU 시간 측정
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
    else:
        start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        # Learning rate scheduling
        # scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

    # 시간 측정 완료
    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()  # GPU 작업 완료 대기
        total_time = start_event.elapsed_time(end_event) / 1000  # 밀리초를 초로 변환
    else:
        total_time = time.time() - start_time
    
    print(f"\n총 학습 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
    
    return train_losses, val_losses, total_time

def evaluate_model(model, test_loader, classes, device):
    """모델 평가 함수"""
    model.eval()
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # Per-class accuracy calculation
            for i in range(y.size(0)):
                label = y[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    overall_accuracy = 100 * correct / total
    class_accuracies = {}
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_accuracies[classes[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[classes[i]] = 0
    
    return overall_accuracy, class_accuracies

def plot_losses(train_losses, val_losses, title):
    """Loss 그래프 그리기"""
    plt.figure(figsize=(20, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{title} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_class_accuracies(class_accuracies, title):
    """클래스별 정확도 그래프 그리기"""
    plt.figure(figsize=(12, 6))
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.bar(classes, accuracies)
    plt.title(f'{title} - Per-Class Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # 각 막대 위에 정확도 값 표시
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data preparation with augmentation
    # Training용 강한 augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),  # 50% 확률로 좌우 반전
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15)    # ±15도 회전
        ], p=0.2),
        transforms.RandomApply([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1))  # 무작위 크기 조정 후 자르기
        ], p=0.2),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,    # 밝기 ±20%
                contrast=0.2,      # 대비 ±20%
                saturation=0.2,    # 채도 ±20%
                hue=0.1           # 색조 ±10%
            )
        ],p=0.2),
        transforms.RandomGrayscale(p=0.1),  # 10% 확률로 흑백 변환
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # 가우시안 블러
        ],p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 10% 확률로 영역 지우기
    ])
    
    # Validation/Test용 기본 변환 (augmentation 없음)
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 데이터셋 로드 (training은 augmentation 적용)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_test_transform)

    classes = trainset.classes
    print(f"Classes: {classes}")

    # Train/Validation split
    torch.manual_seed(43)
    val_size = 5000
    train_size = len(trainset) - val_size
    train_ds, val_ds_temp = random_split(trainset, [train_size, val_size])
    
    # Validation set에는 augmentation을 적용하지 않음
    # 따라서 별도로 validation transform을 적용한 dataset 생성
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=val_test_transform)
    val_indices = val_ds_temp.indices
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)
    
    print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")
    print(f"Data augmentation applied to training set:")
    print("- Random Horizontal Flip (20%)")
    print("- Random Rotation (±15°) (20%)")
    print("- Random Resized Crop (20%)")
    print("- Color Jitter (brightness, contrast, saturation, hue) (20%)")
    print("- Random Grayscale (10%)")
    print("- Gaussian Blur (20%)")
    print("- Random Erasing (10%)")

    # Data loaders
    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size*2, num_workers=4, pin_memory=True)

    num_epochs = 10
    base_learning_rate = 0.0006
    num_classes = 10

    # 0. ResNet-50 Scratch Training
    print("\n" + "="*50)
    print("0. ResNet-50 Scratch Training")
    print("="*50)

    resnet50_scratch = resnet50(pretrained=False)    
    num_ftrs = resnet50_scratch.fc.in_features
    resnet50_scratch.fc = nn.Linear(num_ftrs, num_classes)
    resnet50_scratch.to(device)
    

    # 모든 파라미터 고정 (freeze)
    for param in resnet50_scratch.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer_scratch = optim.Adam(filter(lambda p: p.requires_grad, resnet50_scratch.parameters()), lr=base_learning_rate)
    scheduler_scratch = optim.lr_scheduler.StepLR(optimizer_scratch, step_size=4, gamma=0.1)
    
    train_losses_scratch, val_losses_scratch, total_time_scratch = train_model(
        resnet50_scratch, train_loader, val_loader, criterion, optimizer_scratch, scheduler_scratch, num_epochs, device
    )
    
    # 모델 저장
    torch.save(resnet50_scratch.state_dict(), 'resnet50_scratch.pth')
    print("Model saved as 'resnet50_scratch.pth'")
    
    # 평가
    overall_acc_scratch, class_acc_scratch = evaluate_model(resnet50_scratch, test_loader, classes, device)
    print(f"Scratch Training - Overall Test Accuracy: {overall_acc_scratch:.2f}%")
    
    # 그래프 그리기
    plot_losses(train_losses_scratch, val_losses_scratch, "scratch Training")
    plot_class_accuracies(class_acc_scratch, "scratch Training")

    # 1. ResNet-50 FC layer만 weight 학습 (Classifier Training)
    print("\n" + "="*50)
    print("1. ResNet-50 Classifier Training (FC layer only)")
    print("="*50)
    
    resnet50_classifier = resnet50(pretrained=True)
    num_ftrs = resnet50_classifier.fc.in_features
    resnet50_classifier.fc = nn.Linear(num_ftrs, num_classes)
    resnet50_classifier.to(device)
    
    # 모든 파라미터 고정 (freeze)
    for param in resnet50_classifier.parameters():
        param.requires_grad = False
    
    # FC layer만 학습 가능
    for param in resnet50_classifier.fc.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(filter(lambda p: p.requires_grad, resnet50_classifier.parameters()), lr=base_learning_rate)
    scheduler_classifier = optim.lr_scheduler.StepLR(optimizer_classifier, step_size=4, gamma=0.1)
    
    train_losses_classifier, val_losses_classifier, total_time_classifier = train_model(
        resnet50_classifier, train_loader, val_loader, criterion, optimizer_classifier, scheduler_classifier, num_epochs, device
    )
    
    # 모델 저장
    torch.save(resnet50_classifier.state_dict(), 'resnet50_classifier.pth')
    print("Model saved as 'resnet50_classifier.pth'")
    
    # 평가
    overall_acc_classifier, class_acc_classifier = evaluate_model(resnet50_classifier, test_loader, classes, device)
    print(f"Classifier Training - Overall Test Accuracy: {overall_acc_classifier:.2f}%")
    
    # 그래프 그리기
    plot_losses(train_losses_classifier, val_losses_classifier, "Classifier Training")
    plot_class_accuracies(class_acc_classifier, "Classifier Training")

    # 2. ResNet-50 모든 weight 학습 (Whole Training)
    print("\n" + "="*50)
    print("2. ResNet-50 Whole Training (All layers)")
    print("="*50)
    
    resnet50_whole_layer = resnet50(pretrained=True)
    num_ftrs = resnet50_whole_layer.fc.in_features
    resnet50_whole_layer.fc = nn.Linear(num_ftrs, num_classes)
    resnet50_whole_layer.to(device)
    
    # 모든 파라미터 학습 가능
    for param in resnet50_whole_layer.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer_whole_layer = optim.Adam(resnet50_whole_layer.parameters(), lr=base_learning_rate)
    scheduler_whole_layer = optim.lr_scheduler.StepLR(optimizer_whole_layer, step_size=4, gamma=0.1)
    
    train_losses_whole_layer, val_losses_whole_layer, total_time_whole = train_model(
        resnet50_whole_layer, train_loader, val_loader, criterion, optimizer_whole_layer, scheduler_whole_layer, num_epochs, device
    )
    
    # 모델 저장
    torch.save(resnet50_whole_layer.state_dict(), 'resnet50_whole_layer.pth')
    print("Model saved as 'resnet50_whole_layer.pth'")
    
    # 평가
    overall_acc_whole_layer, class_acc_whole_layer = evaluate_model(resnet50_whole_layer, test_loader, classes, device)
    print(f"Whole_layer Training - Overall Test Accuracy: {overall_acc_whole_layer:.2f}%")
    
    # 그래프 그리기
    plot_losses(train_losses_whole_layer, val_losses_whole_layer, "whole_layer Training")
    plot_class_accuracies(class_acc_whole_layer, "whole_layer Training")

    # 3. ResNet-50 Layer 4 재학습 (last Layers Fine-tuning)
    print("\n" + "="*50)
    print("3. ResNet-50 last Layers Fine-tuning (Layer 4 + FC)")
    print("="*50)
    
    resnet50_last_layers = resnet50(pretrained=True)
    num_ftrs = resnet50_last_layers.fc.in_features
    resnet50_last_layers.fc = nn.Linear(num_ftrs, num_classes)
    resnet50_last_layers.to(device)
    
    # 모든 파라미터 고정
    for param in resnet50_last_layers.parameters():
        param.requires_grad = False
    
    # Layer 4, fc layer만 학습 가능
    for param in resnet50_last_layers.layer4.parameters():
        param.requires_grad = True
    for param in resnet50_last_layers.fc.parameters():
        param.requires_grad = True
    
    # 초기 레이어는 더 높은 학습률 사용
    optimizer_last = optim.Adam(filter(lambda p: p.requires_grad, resnet50_last_layers.parameters()), lr=base_learning_rate)
    scheduler_last = optim.lr_scheduler.StepLR(optimizer_last, step_size=4, gamma=0.1)
    
    train_losses_last, val_losses_last, total_time_last = train_model(
        resnet50_last_layers, train_loader, val_loader, criterion, optimizer_last, scheduler_last, num_epochs, device
    )
    
    # 모델 저장
    torch.save(resnet50_last_layers.state_dict(), 'resnet50_last_layer.pth')
    print("Model saved as 'resnet50_last_layer.pth'")
    
    # 평가
    overall_acc_last, class_acc_last = evaluate_model(resnet50_last_layers, test_loader, classes, device)
    print(f"Last Layers Fine-tuning - Overall Test Accuracy: {overall_acc_last:.2f}%")
    
    # 그래프 그리기
    plot_losses(train_losses_last, val_losses_last, "Last Layers Fine-tuning")
    plot_class_accuracies(class_acc_last, "Last Layers Fine-tuning")

    # 4. ResNet-50 Layer 3,4 재학습 (Late Layers Fine-tuning)
    print("\n" + "="*50)
    print("4. ResNet-50 Late Layers Fine-tuning (Layer 3,4 + FC)")
    print("="*50)
    
    resnet50_late_layers = resnet50(pretrained=True)
    num_ftrs = resnet50_late_layers.fc.in_features
    resnet50_late_layers.fc = nn.Linear(num_ftrs, num_classes)
    resnet50_late_layers.to(device)
    
    # 모든 파라미터 고정
    for param in resnet50_late_layers.parameters():
        param.requires_grad = False
    
    # Layer 3, 4와 fc layer만 학습 가능
    for param in resnet50_late_layers.layer3.parameters():
        param.requires_grad = True
    for param in resnet50_late_layers.layer4.parameters():
        param.requires_grad = True
    for param in resnet50_late_layers.fc.parameters():
        param.requires_grad = True
    
    optimizer_late = optim.Adam(filter(lambda p: p.requires_grad, resnet50_late_layers.parameters()), lr=base_learning_rate)
    scheduler_late = optim.lr_scheduler.StepLR(optimizer_late, step_size=4, gamma=0.1)
    
    train_losses_late, val_losses_late, total_time_late = train_model(
        resnet50_late_layers, train_loader, val_loader, criterion, optimizer_late, scheduler_late, num_epochs, device
    )
    
    # 모델 저장
    torch.save(resnet50_late_layers.state_dict(), 'resnet50_late_layer.pth')
    print("Model saved as 'resnet50_last_layer.pth'")
    
    # 평가
    overall_acc_late, class_acc_late = evaluate_model(resnet50_late_layers, test_loader, classes, device)
    print(f"Late Layers Fine-tuning - Overall Test Accuracy: {overall_acc_late:.2f}%")
    
    # 그래프 그리기
    plot_losses(train_losses_late, val_losses_late, "Late Layers Fine-tuning")
    plot_class_accuracies(class_acc_late, "Late Layers Fine-tuning")

    # 결과 요약
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    print(f"Scratch Training - Overall Accuracy: {overall_acc_scratch:.2f}%")
    print(f"Classifier Training (FC only) - Overall Accuracy: {overall_acc_classifier:.2f}%")
    print(f"Whole Training - Overall Accuracy: {overall_acc_whole_layer:.2f}%")
    print(f"Last Layers Fine-tuning (Layer 4) - Overall Accuracy: {overall_acc_last:.2f}%")
    print(f"Late Layers Fine-tuning (Layer 3,4) - Overall Accuracy: {overall_acc_late:.2f}%")
    
    print(f"\nLearning Rates Used:")
    print(f"Classifier & Scratch Training: {base_learning_rate}")
    # print(f"last Layers (Layer 1,2): {base_learning_rate} (higher LR)")
    # print(f"Late Layers (Layer 3,4): {base_learning_rate} (lower LR to prevent overshooting)")
    
    # 전체 결과 비교 그래프
    plt.figure(figsize=(15, 5))
    
    # Loss 비교
    plt.subplot(1, 2, 1)
    epochs = range(1, num_epochs + 1)    
    plt.plot(epochs, train_losses_scratch, 'm-', label='Scratch - Train')
    plt.plot(epochs, val_losses_scratch, 'm--', label='Scratch - Val')
    plt.plot(epochs, train_losses_classifier, 'k-', label='Classifier - Train')
    plt.plot(epochs, val_losses_classifier, 'k--', label='Classifier - Val')
    plt.plot(epochs, train_losses_last, 'b-', label='Last Layers - Train')
    plt.plot(epochs, val_losses_last, 'b--', label='Last Layers - Val')
    plt.plot(epochs, train_losses_late, 'r-', label='Late Layers - Train')
    plt.plot(epochs, val_losses_late, 'r--', label='Late Layers - Val')
    plt.plot(epochs, train_losses_whole_layer, 'g-', label='Whole - Train')
    plt.plot(epochs, val_losses_whole_layer, 'g--', label='Whole - Val')
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 전체 정확도 비교
    plt.subplot(1, 2, 2)
    methods = ['Scratch\nTraining','Classifier\n(FC only)', 'Last Layers\n(Layer 4)', 'Late Layers\n(Layer 3,4)','Whole Layers\nTraining']
    accuracies = [overall_acc_scratch,overall_acc_classifier, overall_acc_last, overall_acc_late,overall_acc_whole_layer]
    colors = ['purple','black', 'blue', 'red', 'green']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.title('Overall Test Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show() 

    # 전체 시간 비교
    plt.figure(figsize=(12, 6))

    # 시간 비교 그래프
    plt.subplot(1, 2, 1)
    methods = ['Scratch\nTraining','Classifier\n(FC only)','Last Layers\n(Layer 4)',  'Late Layers\n(Layer 3,4)', 'Whole Layers\nTraining']
    total_times = [total_time_scratch, total_time_classifier, total_time_last, total_time_late, total_time_whole]
    colors = ['purple','black', 'blue', 'red', 'green']

    bars = plt.bar(methods, total_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontsize=12)

    # y축 범위를 데이터에 맞게 자동 조정하되, 여유 공간 확보
    max_time = max(total_times)
    plt.ylim(0, max_time * 1.15)

    # 막대 위에 시간 값 표시 (초와 분 단위로)
    for bar, time_val in zip(bars, total_times):
        height = bar.get_height()
        # 시간이 60초 이상이면 분:초 형식으로 표시
        if time_val >= 60:
            minutes = int(time_val // 60)
            seconds = int(time_val % 60)
            time_text = f'{time_val:.1f}s\n({minutes}:{seconds:02d})'
        else:
            time_text = f'{time_val:.1f}s'
        
        plt.text(bar.get_x() + bar.get_width()/2, height + max_time * 0.02, 
                time_text, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 격자 추가로 가독성 향상
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')

    # 상대적 시간 비교 (정규화된 막대 그래프)
    plt.subplot(1, 2, 2)
    # 가장 빠른 시간을 기준으로 정규화
    min_time = min(total_times)
    normalized_times = [t / min_time for t in total_times]

    bars2 = plt.bar(methods, normalized_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.title('Relative Training Time\n(Normalized to Fastest)', fontsize=14, fontweight='bold')
    plt.ylabel('Relative Time (×)', fontsize=12)
    plt.ylim(0, max(normalized_times) * 1.15)

    # 상대적 시간 표시
    for bar, norm_time, abs_time in zip(bars2, normalized_times, total_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(normalized_times) * 0.02, 
                f'{norm_time:.1f}×\n({abs_time:.1f}s)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.0x 기준선 추가
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (Fastest)')
    plt.legend()

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # 시간 차이 통계 출력
    print("=== Training Time Analysis ===")
    print(f"Fastest method: {methods[np.argmin(total_times)]} ({min(total_times):.1f}s)")
    print(f"Slowest method: {methods[np.argmax(total_times)]} ({max(total_times):.1f}s)")
    print(f"Speed difference: {max(total_times)/min(total_times):.1f}× slower")
    print(f"Time saved by fastest: {max(total_times) - min(total_times):.1f}s ({(max(total_times) - min(total_times))/60:.1f} min)")

    print("\nDetailed comparison:")
    for method, time_val in zip(methods, total_times):
        relative = time_val / min_time
        print(f"{method.replace(chr(10), ' ')}: {time_val:.1f}s ({relative:.1f}× baseline)")