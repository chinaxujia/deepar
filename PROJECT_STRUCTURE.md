# 项目目录结构说明

## 📁 目录结构

```
deepar/
├── app/                          # 应用程序主目录
│   ├── __init__.py              # 应用包初始化
│   ├── api/                     # API路由模块
│   │   ├── __init__.py
│   │   └── v1/                  # API v1版本
│   │       ├── __init__.py
│   │       ├── health.py        # 健康检查API
│   │       ├── training.py      # 模型训练API
│   │       ├── prediction.py    # 模型预测API
│   │       ├── data.py          # 数据管理API
│   │       ├── models.py        # 模型管理API
│   │       └── monitoring.py    # 系统监控API
│   ├── core/                    # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py            # 应用配置
│   │   └── logging.py           # 日志配置
│   ├── models/                  # 数据模型定义
│   │   └── __init__.py
│   ├── schemas/                 # Pydantic数据模型
│   │   ├── __init__.py
│   │   └── health.py            # 健康检查Schema
│   ├── services/                # 业务服务层
│   │   ├── __init__.py
│   │   └── redis_service.py     # Redis服务
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       └── helpers.py           # 辅助函数
├── configs/                     # 配置文件目录
│   └── model_config.json        # 模型配置示例
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   │   └── sample_energy_data.csv
│   └── processed/               # 处理后的数据
├── models/                      # 训练好的模型文件
├── logs/                        # 日志文件目录
├── notebooks/                   # Jupyter笔记本
├── scripts/                     # 脚本目录
├── app.py                       # FastAPI应用入口
├── requirements.txt             # Python依赖
├── docker-compose.yml           # Docker Compose配置
├── docker-compose.dev.yml       # 开发环境Docker配置
├── Dockerfile                   # Docker镜像定义
├── .env.example                 # 环境变量示例
├── .dockerignore               # Docker忽略文件
├── start.sh                    # Linux/macOS启动脚本
├── start.ps1                   # Windows启动脚本
├── DOCKER.md                   # Docker部署指南
└── readme.md                   # 项目文档
```

## 📝 文件说明

### 🚀 应用程序文件
- **`app.py`**: FastAPI应用程序主入口文件
- **`app/`**: 应用程序包目录，包含所有业务逻辑

### ⚙️ 配置文件
- **`.env.example`**: 环境变量配置模板
- **`configs/`**: 应用配置文件目录，JSON格式
- **`app/core/config.py`**: 应用程序配置管理

### 🔌 API模块
- **`app/api/v1/`**: API v1版本路由定义
- **`app/schemas/`**: Pydantic数据验证模型
- **`app/services/`**: 业务服务层，处理核心业务逻辑

### 💾 数据存储
- **`data/raw/`**: 原始训练数据存储
- **`data/processed/`**: 预处理后的数据
- **`models/`**: 训练完成的模型文件
- **`logs/`**: 应用程序日志文件

### 🧪 开发和调试
- **`notebooks/`**: Jupyter笔记本，用于数据分析和原型开发
- **`scripts/`**: 辅助脚本

### 🐳 部署文件
- **`Dockerfile`**: Docker镜像构建文件
- **`docker-compose.yml`**: 生产环境Docker编排
- **`docker-compose.dev.yml`**: 开发环境Docker编排
- **`start.sh/.ps1`**: 便捷启动脚本

## 🔧 开发指南

### 添加新的API端点
1. 在 `app/api/v1/` 下创建或修改路由文件
2. 在 `app/schemas/` 中定义请求/响应模型
3. 在 `app/services/` 中实现业务逻辑
4. 在 `app.py` 中注册新路由

### 添加新的服务
1. 在 `app/services/` 下创建服务类
2. 在 `app/core/config.py` 中添加相关配置
3. 在需要的地方注入服务依赖

### 数据模型管理
- 使用 `app/models/` 定义数据库模型
- 使用 `app/schemas/` 定义API输入输出模型
- 配置文件放在 `configs/` 目录

### 日志和监控
- 日志配置在 `app/core/logging.py`
- 系统监控API在 `app/api/v1/monitoring.py`
- 健康检查API在 `app/api/v1/health.py`

## 🎯 下一步开发计划
1. 实现完整的训练API
2. 实现预测服务API
3. 添加数据验证和预处理
4. 实现模型管理功能
5. 完善监控和告警