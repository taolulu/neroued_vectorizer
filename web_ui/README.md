# neroued-vectorizer Web UI

Gradio 浏览器界面，基于 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer) C++ 库实现高质量栅格到 SVG 矢量化。

![screenshot](https://raw.githubusercontent.com/taolulu/neroued_vectorizer/add-gradio-web-ui/docs/web_ui_screenshot.png)

## 功能

- 上传图片，浏览器内完成矢量化，无需命令行
- 调参面板：颜色数量、曲线拟合误差、平滑度、角落阈值、子像素细化等
- SVG 实时预览 + 一键下载

## 快速开始

### Docker（推荐）

```bash
cd web_ui
docker compose up --build
```

服务启动后访问 http://localhost:7861

### 手动运行

```bash
pip install -r requirements.txt
python gradio_app.py
```

依赖 Python >= 3.11，依赖 `neroued-vectorizer` pip 包（自动从 PyPI 安装）。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Number of Colors | 0 (auto) | 调色板颜色数量，0 表示自动 |
| Min Region Area | 0 | 最小区域面积（px²），0 保留全部 |
| Curve Fit Error | 1.0 | 贝塞尔曲线拟合误差阈值（px） |
| Corner Angle Threshold | 60° | 角落检测角度阈值 |
| Smoothness | 0.3 | 路径平滑强度 [0, 1] |
| Smoothing Spatial Radius | 10.0 | 空间平滑半径 |
| Smoothing Color Radius | 10.0 | 颜色平滑半径 |
| Upscale Short Edge | 0 (off) | 短边放大尺寸（提高精度） |
| Max Working Pixels | 4M | 最大处理像素数（0=无限） |
| Enable Sub-pixel Refinement | True | 亚像素边界细化 |

## 项目结构

```
web_ui/
├── gradio_app.py          # Gradio Web UI 实现
├── requirements.txt       # Python 依赖
├── Dockerfile              # 容器构建文件
├── docker-compose.yml      # Docker Compose 配置
└── README.md               # 本文件
```

## 与上游关系

本目录是 [neroued/neroued_vectorizer](https://github.com/neroued/neroued_vectorizer) 的附加组件，依赖上游发布的 `neroued-vectorizer` pip 包。

- 上游仓库：https://github.com/neroued/neroued_vectorizer
- 上游许可证：GPL-3.0-or-later

## License

GPL-3.0-or-later（与上游一致）
