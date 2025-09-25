#!/bin/bash

# DeepAR 能耗预测服务启动脚本

echo "=== DeepAR 工厂能耗预测服务 ==="
echo "启动时间: $(date)"
echo

# 检查Docker是否已安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查docker-compose是否已安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: docker-compose 未安装，请先安装 docker-compose"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p models configs data logs

# 复制环境配置文件
if [ ! -f .env ]; then
    echo "创建环境配置文件..."
    cp .env.example .env
    echo "请编辑 .env 文件以配置您的环境"
fi

# 函数：启动服务
start_services() {
    echo "启动 DeepAR 预测服务..."
    docker-compose up -d
    
    echo "等待服务启动..."
    sleep 10
    
    # 检查服务状态
    echo "检查服务状态:"
    docker-compose ps
    
    echo
    echo "服务访问地址:"
    echo "  - API 服务: http://localhost:8000"
    echo "  - API 文档: http://localhost:8000/docs"
    echo "  - Redis 管理: http://localhost:8081 (可选)"
    echo
    echo "查看日志: docker-compose logs -f"
    echo "停止服务: docker-compose down"
}

# 函数：停止服务
stop_services() {
    echo "停止 DeepAR 预测服务..."
    docker-compose down
    echo "服务已停止"
}

# 函数：重启服务
restart_services() {
    echo "重启 DeepAR 预测服务..."
    docker-compose restart
    echo "服务已重启"
}

# 函数：查看日志
show_logs() {
    docker-compose logs -f
}

# 函数：显示帮助
show_help() {
    echo "用法: $0 [COMMAND]"
    echo
    echo "命令:"
    echo "  start     启动服务"
    echo "  stop      停止服务"
    echo "  restart   重启服务"
    echo "  logs      查看日志"
    echo "  status    查看服务状态"
    echo "  help      显示帮助"
    echo
}

# 函数：查看服务状态
show_status() {
    echo "服务状态:"
    docker-compose ps
    echo
    echo "服务健康检查:"
    curl -s http://localhost:8000/health || echo "API 服务不可用"
}

# 主逻辑
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    help)
        show_help
        ;;
    *)
        echo "未知命令: $1"
        show_help
        exit 1
        ;;
esac