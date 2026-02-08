"""
项目：基于轻量级Transformer的手写文本序列识别
文件：train.py
作者：林泽远
日期：2026.02
功能：训练Mini Transformer在MNIST数据集上的分类任务，记录实验数据
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mini_transformer import MiniTransformer, MiniTransformerConfig

# ====================== 1. 日志配置（科研项目必备，替代print） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 打印到控制台
)
logger = logging.getLogger(__name__)

# ====================== 2. 全局配置（统一管理，易调参） ======================
class TrainConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32          # 批次大小（轻量化）
        self.epochs = 2               # 训练轮数（先跑通）
        self.lr = 1e-3                # 学习率
        self.data_dir = "./data"      # 数据存放路径
        self.save_dir = "./checkpoints"  # 模型保存路径
        self.model_name = "mini_transformer_best.pth"  # 模型文件名

# ====================== 3. 数据加载（复用CNN逻辑，标准化） ======================
def load_mnist_data(config: TrainConfig):
    """加载MNIST数据集，返回训练/测试加载器"""
    # 数据预处理（和CNN一致，保证对比公平）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST固定标准化参数
    ])
    
    # 下载/加载数据集
    train_dataset = datasets.MNIST(
        root=config.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir, train=False, download=True, transform=transform
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"✅ 数据集加载完成！训练集: {len(train_dataset)} 样本，测试集: {len(test_dataset)} 样本")
    return train_loader, test_loader

# ====================== 4. 训练/测试函数（模块化，易复用） ======================
def train_one_epoch(model, train_loader, optimizer, criterion, config: TrainConfig, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config.device), target.to(config.device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播 + 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        # 每100批次打印日志
        if batch_idx % 100 == 0:
            batch_acc = 100. * correct / total_samples
            logger.info(f"Epoch {epoch} [{batch_idx*config.batch_size}/{len(train_loader.dataset)}] "
                        f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%")
    
    # 计算epoch级指标
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = 100. * correct / len(train_loader.dataset)
    logger.info(f"Epoch {epoch} Train - Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%")
    return avg_loss, avg_acc

def evaluate(model, test_loader, criterion, config: TrainConfig):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    
    with torch.no_grad():  # 关闭梯度，节省显存
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # 计算评估指标
    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = 100. * correct / len(test_loader.dataset)
    logger.info(f"Test - Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%\n")
    return avg_loss, avg_acc

# ====================== 5. 主训练流程（入口函数） ======================
def main():
    # 1. 初始化配置
    train_config = TrainConfig()
    model_config = MiniTransformerConfig(
        img_size=28,
        embed_dim=64,
        num_heads=2,
        num_layers=1
    )
    
    # 2. 创建保存目录
    os.makedirs(train_config.save_dir, exist_ok=True)
    
    # 3. 加载数据
    train_loader, test_loader = load_mnist_data(train_config)
    
    # 4. 初始化模型、损失、优化器
    model = MiniTransformer(model_config).to(train_config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
    
    # 打印模型信息（科研调试必备）
    logger.info(f"✅ 模型初始化成功！设备: {train_config.device}")
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())/1000:.2f}k")
    
    # 5. 启动训练
    best_acc = 0.0
    for epoch in range(1, train_config.epochs + 1):
        # 训练
        train_one_epoch(model, train_loader, optimizer, criterion, train_config, epoch)
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, train_config)
        
        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(train_config.save_dir, train_config.model_name))
            logger.info(f"📌 保存最优模型！准确率: {best_acc:.2f}%")
    
    # 训练完成
    logger.info(f"🎉 训练结束！最优测试准确率: {best_acc:.2f}%")
    return best_acc

# ====================== 6. 运行入口 ======================
if __name__ == "__main__":
    main()  