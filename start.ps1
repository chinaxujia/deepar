# DeepAR 能耗预测服务启动脚本 (PowerShell)

param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "restart", "logs", "status", "help")]
    [string]$Command = "start"
)

Write-Host "=== DeepAR 工厂能耗预测服务 ===" -ForegroundColor Green
Write-Host "启动时间: $(Get-Date)" -ForegroundColor Gray
Write-Host ""

# 检查Docker是否已安装
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "错误: Docker 未安装，请先安装 Docker Desktop" -ForegroundColor Red
    exit 1
}

# 检查docker-compose是否已安装
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "错误: docker-compose 未安装，请先安装 docker-compose" -ForegroundColor Red
    exit 1
}

# 创建必要的目录
Write-Host "创建必要的目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path models, configs, data, logs | Out-Null

# 复制环境配置文件
if (-not (Test-Path .env)) {
    Write-Host "创建环境配置文件..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "请编辑 .env 文件以配置您的环境" -ForegroundColor Cyan
}

# 函数：启动服务
function Start-Services {
    Write-Host "启动 DeepAR 预测服务..." -ForegroundColor Green
    docker-compose up -d
    
    Write-Host "等待服务启动..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # 检查服务状态
    Write-Host "检查服务状态:" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host ""
    Write-Host "服务访问地址:" -ForegroundColor Green
    Write-Host "  - API 服务: http://localhost:8000" -ForegroundColor White
    Write-Host "  - API 文档: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  - Redis 管理: http://localhost:8081 (可选)" -ForegroundColor White
    Write-Host ""
    Write-Host "查看日志: docker-compose logs -f" -ForegroundColor Gray
    Write-Host "停止服务: docker-compose down" -ForegroundColor Gray
}

# 函数：停止服务
function Stop-Services {
    Write-Host "停止 DeepAR 预测服务..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "服务已停止" -ForegroundColor Green
}

# 函数：重启服务
function Restart-Services {
    Write-Host "重启 DeepAR 预测服务..." -ForegroundColor Yellow
    docker-compose restart
    Write-Host "服务已重启" -ForegroundColor Green
}

# 函数：查看日志
function Show-Logs {
    docker-compose logs -f
}

# 函数：显示帮助
function Show-Help {
    Write-Host "用法: .\start.ps1 [COMMAND]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "命令:" -ForegroundColor Yellow
    Write-Host "  start     启动服务" -ForegroundColor White
    Write-Host "  stop      停止服务" -ForegroundColor White
    Write-Host "  restart   重启服务" -ForegroundColor White
    Write-Host "  logs      查看日志" -ForegroundColor White
    Write-Host "  status    查看服务状态" -ForegroundColor White
    Write-Host "  help      显示帮助" -ForegroundColor White
    Write-Host ""
}

# 函数：查看服务状态
function Show-Status {
    Write-Host "服务状态:" -ForegroundColor Cyan
    docker-compose ps
    Write-Host ""
    Write-Host "服务健康检查:" -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "API 服务正常" -ForegroundColor Green
    }
    catch {
        Write-Host "API 服务不可用" -ForegroundColor Red
    }
}

# 主逻辑
switch ($Command) {
    "start" { Start-Services }
    "stop" { Stop-Services }
    "restart" { Restart-Services }
    "logs" { Show-Logs }
    "status" { Show-Status }
    "help" { Show-Help }
    default {
        Write-Host "未知命令: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}