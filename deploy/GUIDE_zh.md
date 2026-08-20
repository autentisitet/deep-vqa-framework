# 部署指南

English version: [deploy/GUIDE.md](GUIDE.md)

本指南集中说明 Deep-VQA 推理服务、前端和批量 CLI 的运行方式。

## 1. 前置条件

请从项目根目录确认模型检查点存在：

```text
deploy/iqa-models/tid2013_best.pt
deploy/vqa-models/konvid-1k_best.pt
```

服务会从 checkpoint 中读取模型配置和 MOS 范围。如果两个 checkpoint
都不存在，服务会主动启动失败。

在再分发或商业部署 checkpoint 前，请先阅读[免责声明与资源使用政策](../DISCLAIMER_zh.md)。

## 2. 选择运行模式

| 模式 | 命令 | 适用场景 | 入口 |
| --- | --- | --- | --- |
| 主机 API | `uv run python -m uvicorn deploy.api:app --host 127.0.0.1 --port 8000` | 本地调试模型和 API | `/v1/*` |
| 容器 API | `make docker-api` | 只调试容器中的 FastAPI | `http://127.0.0.1:8001/v1/*` |
| 完整服务 | `make docker-infer` | 前端、Nginx 和 API 一起演示或交付 | `http://127.0.0.1:8000/`、`/api/v1/*` |

`docker-api` 只启动 FastAPI；`docker-infer` 会构建镜像、等待 FastAPI 健康、启动 Nginx，并自动验证代理后的健康接口。

## 3. 主机直运行

```bash
uv run python -m uvicorn deploy.api:app --host 127.0.0.1 --port 8000
```

直接访问：

```text
健康检查：http://127.0.0.1:8000/v1/health
模型列表：http://127.0.0.1:8000/v1/models
Swagger：http://127.0.0.1:8000/docs
ReDoc：http://127.0.0.1:8000/redoc
```

主机直运行不会托管静态前端。如需预览前端，可另开终端运行：

```bash
python -m http.server 5500 -d frontend
```

## 4. 容器运行

仅启动 API：

```bash
make docker-api
curl --noproxy '*' http://127.0.0.1:8001/v1/health
```

启动完整前端和 API：

```bash
make docker-infer
curl --noproxy '*' http://127.0.0.1:8000/api/v1/health
```

完整链路为：

```text
浏览器 :8000 → Nginx → vqa-infer:8000 → FastAPI → IQA/VQA checkpoint
```

## 5. Docker 与 Podman

Makefile 会自动检测可用的容器运行时，不需要创建 `docker=podman` 别名。

Podman overlay 会为 SELinux 主机上的 Nginx 挂载添加 `:Z`；Ubuntu 等未启用 SELinux 的 Podman 环境也可以使用同一套命令。

## 6. API 接口

经过 Nginx 时使用 `/api` 前缀：

```text
GET  /api/v1/health
GET  /api/v1/models
GET  /api/v1/models/{model_id}
POST /api/v1/evaluations?model_id=iqa|vqa
POST /api/v1/visualizations?model_id=iqa
```

评估接口接收一个名为 `file` 的 multipart 文件。可视化接口只接受图像，生成的特征图和 Grad-CAM 会保存到 `results/diagnostics/`。

主机直运行或 API-only 模式不使用 `/api` 前缀。

## 7. 批量推理

```bash
uv run python -m deploy.cli -i examples/images/
uv run python -m deploy.cli -i examples/videos/
uv run python -m deploy.cli -i examples/images/sample.jpg --visualize
```

## 8. 代理、SELinux 与排障

Compose 会传入大小写两种代理变量，并将 localhost 和 Compose 服务名加入 `NO_PROXY`/`no_proxy`，避免健康检查和容器间请求被宿主机代理拦截。Fedora/RHEL 上如果 Nginx 无法读取 `default.conf`，重新创建容器以应用 Podman 的 `:Z` 挂载：

```bash
podman-compose -f docker/docker-compose.yaml -f docker/docker-compose.podman.yaml down
make docker-infer
```

常用排障命令：

```bash
podman ps -a
podman logs vqa-infer
podman logs vqa-nginx
podman inspect vqa-infer --format '{{.State.Status}} {{.State.Health.Status}}'
curl --noproxy '*' -v http://127.0.0.1:8000/api/v1/health
```
