# Docker 部署指南

## 快速开始

### 1. 准备环境

确保已安装以下软件：
- Docker Desktop (Windows/macOS) 或 Docker Engine (Linux)
- Docker Compose

### 2. 配置环境变量

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件（可选）
# Windows: notepad .env
# Linux/macOS: nano .env
```

### 3. 启动服务

#### 方式一：使用启动脚本（推荐）

**Windows PowerShell:**
```powershell
.\start.ps1 start
```

**Linux/macOS:**
```bash
chmod +x start.sh
./start.sh start
```

#### 方式二：直接使用 Docker Compose

```bash
# 启动基础服务
docker-compose up -d

# 启动开发环境（包含更多工具）
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 4. 访问服务

- **API 服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **Redis 管理**: http://localhost:8081 (可选)

## 开发环境

### 启动开发环境

```bash
# 启动完整开发环境
docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev up -d
```

开发环境包含：
- **API 服务**: 支持热重载
- **Redis**: 数据缓存
- **Redis Commander**: Redis 管理界面
- **Jupyter Notebook**: 数据分析和模型开发
  - 访问地址: http://localhost:8888
  - Token: deepar2024

### 启用监控工具

```bash
# 启动监控工具
docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile monitoring up -d
```

监控工具：
- **Portainer**: Docker 容器管理
  - 访问地址: http://localhost:9000

## 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f deepar-api
```

### 服务调试

```bash
# 进入API容器
docker-compose exec deepar-api bash

# 进入Redis容器
docker-compose exec redis redis-cli

# 重建镜像
docker-compose build --no-cache deepar-api
```

### 数据管理

```bash
# 备份Redis数据
docker-compose exec redis redis-cli BGSAVE

# 清理所有数据
docker-compose down -v
```

## 目录结构

```
deepar/
├── docker-compose.yml          # 生产环境配置
├── docker-compose.dev.yml      # 开发环境配置
├── Dockerfile                  # API服务镜像
├── requirements.txt            # Python依赖
├── .env.example               # 环境变量模板
├── .dockerignore              # Docker忽略文件
├── start.sh                   # Linux/macOS启动脚本
├── start.ps1                  # Windows启动脚本
├── models/                    # 模型存储目录
├── configs/                   # 配置文件目录
├── data/                      # 数据文件目录
├── logs/                      # 日志文件目录
└── notebooks/                 # Jupyter笔记本目录
```

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| REDIS_HOST | localhost | Redis主机地址 |
| REDIS_PORT | 6379 | Redis端口 |
| REDIS_DB | 0 | Redis数据库编号 |
| API_PORT | 8000 | API服务端口 |
| LOG_LEVEL | INFO | 日志级别 |
| DEBUG | false | 调试模式 |
| MODEL_STORAGE_PATH | ./models | 模型存储路径 |
| CONFIG_STORAGE_PATH | ./configs | 配置存储路径 |

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 查看端口占用
   netstat -an | grep :8000
   
   # 修改docker-compose.yml中的端口映射
   ports:
     - "8001:8000"  # 改为8001端口
   ```

2. **容器启动失败**
   ```bash
   # 查看详细错误日志
   docker-compose logs deepar-api
   
   # 重建容器
   docker-compose down
   docker-compose up -d --build
   ```

3. **Redis连接失败**
   ```bash
   # 检查Redis状态
   docker-compose exec redis redis-cli ping
   
   # 重启Redis
   docker-compose restart redis
   ```

4. **权限问题（Linux/macOS）**
   ```bash
   # 设置目录权限
   sudo chown -R $USER:$USER models configs data logs
   chmod -R 755 models configs data logs
   ```

### 性能优化

1. **内存限制**
   ```yaml
   # 在docker-compose.yml中添加
   deploy:
     resources:
       limits:
         memory: 2G
       reservations:
         memory: 1G
   ```

2. **CPU限制**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
   ```

## 生产部署建议

1. 使用环境变量管理敏感配置
2. 启用Redis持久化
3. 配置日志轮转
4. 设置资源限制
5. 使用反向代理（Nginx）
6. 启用HTTPS

## 清理资源

```bash
# 停止并删除所有容器
docker-compose down

# 删除所有相关镜像
docker-compose down --rmi all

# 删除所有数据卷
docker-compose down -v

# 清理Docker系统
docker system prune -a
```