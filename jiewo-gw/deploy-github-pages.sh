#!/bin/bash

# 🚀 SolveMe Protocol GitHub Pages 部署脚本
# 使用方法: ./deploy-github-pages.sh

echo "🚀 开始部署 SolveMe Protocol 网站到 GitHub Pages..."

# 检查是否在正确的目录
if [ ! -f "index.html" ]; then
    echo "❌ 错误：请在 jiewo-gw 目录下运行此脚本"
    exit 1
fi

# 检查Git是否已初始化
if [ ! -d ".git" ]; then
    echo "📦 初始化Git仓库..."
    git init
fi

# 添加所有文件
echo "📁 添加文件到Git..."
git add .

# 提交更改
echo "💾 提交更改..."
git commit -m "Update SolveMe Protocol website - $(date)"

# 检查是否已设置远程仓库
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  请先设置GitHub远程仓库："
    echo "git remote add origin https://github.com/YOUR_USERNAME/solveme-protocol-website.git"
    echo "然后运行: git push -u origin main"
    exit 1
fi

# 推送到GitHub
echo "🌐 推送到GitHub..."
git push origin main

echo "✅ 部署完成！"
echo ""
echo "📋 下一步操作："
echo "1. 访问您的GitHub仓库"
echo "2. 进入 Settings → Pages"
echo "3. Source选择: Deploy from a branch"
echo "4. Branch选择: main"
echo "5. 点击 Save"
echo ""
echo "🌍 网站将在5-10分钟后生效"
echo "默认地址: https://YOUR_USERNAME.github.io/solveme-protocol-website" 