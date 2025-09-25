@echo off
:: DeepAR工厂能耗预测服务启动脚本 (Windows版)
:: 作者: DeepAR Team  
:: 版本: 1.0.0

setlocal enabledelayedexpansion

:: 设置标题
title DeepAR工厂能耗预测服务

:: 检查Docker是否安装
echo 🏭 DeepAR工厂能耗预测服务管理脚本 v1.0.0
echo ================================================

docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose未安装，请先安装Docker Compose
    pause
    exit /b 1
)

echo ✅ Docker环境检查通过

:: 创建必要目录
echo 📁 创建必要目录...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "configs" mkdir configs
echo ✅ 目录创建完成

:: 获取操作参数
set action=%1
if "%action%"=="" set action=start

if "%action%"=="start" goto start
if "%action%"=="stop" goto stop
if "%action%"=="restart" goto restart
if "%action%"=="status" goto status
if "%action%"=="logs" goto logs
if "%action%"=="cleanup" goto cleanup
if "%action%"=="help" goto help
goto help

:start
echo 🚀 启动DeepAR预测服务...

:: 停止可能存在的旧容器
docker-compose down >nul 2>&1

:: 构建并启动服务
docker-compose up -d --build
if errorlevel 1 (
    echo ❌ 服务启动失败
    pause
    exit /b 1
)

:: 等待服务启动
echo ⏳ 等待服务启动...
timeout /t 10 /nobreak >nul

:: 检查服务状态
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ 服务启动失败，请查看日志
    docker-compose logs deepar-app
    pause
    exit /b 1
)

echo 🎉 服务启动成功!
echo 📖 API文档: http://localhost:8000/docs
echo 🏠 服务首页: http://localhost:8000
echo ❤️  健康检查: http://localhost:8000/health

:: 显示服务状态
goto status_only

:stop
echo 🛑 停止DeepAR预测服务...
docker-compose down
echo ✅ 服务已停止
goto end

:restart
call :stop
timeout /t 2 /nobreak >nul
goto start

:status
echo 📊 服务状态:
docker-compose ps
echo.
echo 📈 资源使用:
docker stats --no-stream deepar-app deepar-redis 2>nul
goto end

:status_only
echo 📊 服务状态:
docker-compose ps
goto end

:logs
echo 📋 实时日志 (按Ctrl+C退出):
docker-compose logs -f
goto end

:cleanup
echo 🧹 清理Docker环境...
docker-compose down -v --rmi all
docker system prune -f
echo ✅ 清理完成
goto end

:help
echo.
echo 🏭 DeepAR工厂能耗预测服务管理脚本
echo.
echo 用法: %0 [选项]
echo.
echo 选项:
echo     start     启动服务 ^(默认^)
echo     stop      停止服务
echo     restart   重启服务
echo     status    查看服务状态
echo     logs      查看实时日志
echo     cleanup   清理Docker环境
echo     help      显示帮助信息
echo.
echo 示例:
echo     %0              # 启动服务
echo     %0 start        # 启动服务
echo     %0 status       # 查看状态
echo     %0 logs         # 查看日志
echo     %0 stop         # 停止服务
echo     %0 cleanup      # 清理环境
echo.
echo 服务地址:
echo     - API文档: http://localhost:8000/docs
echo     - 服务首页: http://localhost:8000
echo     - 健康检查: http://localhost:8000/health
echo.

:end
pause