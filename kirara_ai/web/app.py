import asyncio
import mimetypes
import os
import socket
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, g, jsonify

from kirara_ai.config.global_config import GlobalConfig
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.logger import HypercornLoggerWrapper, get_logger
from kirara_ai.web.auth.services import AuthService, FileBasedAuthService
from kirara_ai.web.utils import create_no_cache_response, install_webui

from .api.block import block_bp
from .api.dispatch import dispatch_bp
from .api.im import im_bp
from .api.llm import llm_bp
from .api.mcp import mcp_bp
from .api.media import media_bp
from .api.plugin import plugin_bp
from .api.system import system_bp
from .api.tracing import tracing_bp
from .api.workflow import workflow_bp
from .auth.routes import auth_bp

ERROR_MESSAGE = """
<h1>WebUI launch failed!</h1>
<p lang="en">Web UI not found. Please download from <a href='https://github.com/DarkSkyTeam/chatgpt-for-bot-webui/releases' target='_blank'>here</a> and extract to the <span>TARGET_DIR</span> folder, make sure the <span>TARGET_DIR/index.html</span> file exists.</p>
<h1>WebUI 启动失败！</h1>
<p lang="zh-CN">Web UI 未找到。请从 <a href='https://github.com/DarkSkyTeam/chatgpt-for-bot-webui/releases' target='_blank'>这里</a> 下载并解压到 <span>TARGET_DIR</span> 文件夹，确保 <span>TARGET_DIR/index.html</span> 文件存在。</p>

<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
        color: #333;
        padding: 20px;
    }
    h1 {
        color: #333;
        font-size: 24px;
        margin-bottom: 10px;
    }
    p {
        font-size: 16px;
        margin-bottom: 10px;
    }
    a {
        color: #007bff;
        text-decoration: none;
    }
</style>
"""


cwd = os.getcwd()
STATIC_FOLDER = f"{cwd}/web"

logger = get_logger("WebServer")

custom_static_assets: dict[str, str] = {}

def create_web_api_app(container: DependencyContainer) -> Quart:
    """创建 Web API 应用（Quart）"""
    app = Quart(__name__, static_folder=STATIC_FOLDER)
    app.json.sort_keys = False # type: ignore

    # 注册蓝图
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(im_bp, url_prefix="/api/im")
    app.register_blueprint(llm_bp, url_prefix="/api/llm")
    app.register_blueprint(dispatch_bp, url_prefix="/api/dispatch")
    app.register_blueprint(block_bp, url_prefix="/api/block")
    app.register_blueprint(workflow_bp, url_prefix="/api/workflow")
    app.register_blueprint(plugin_bp, url_prefix="/api/plugin")
    app.register_blueprint(system_bp, url_prefix="/api/system")
    app.register_blueprint(media_bp, url_prefix="/api/media")
    app.register_blueprint(tracing_bp, url_prefix="/api/tracing")
    app.register_blueprint(mcp_bp, url_prefix="/api/mcp")

    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.opt(exception=error).error("Error during request")
        response = jsonify({"error": str(error)})
        response.status_code = 500
        return response

    # 在每个请求前将容器注入到上下文
    @app.before_request
    async def inject_container(): # type: ignore
        g.container = container

    @app.before_websocket
    async def inject_container_ws(): # type: ignore
        g.container = container

    app.container = container # type: ignore

    return app

def create_app(container: DependencyContainer) -> FastAPI:
    """创建主应用（FastAPI）"""
    app = FastAPI()

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 强制设置 MIME 类型
    mimetypes.add_type("text/html", ".html")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("image/svg+xml", ".svg")
    mimetypes.add_type("image/png", ".png")
    mimetypes.add_type("image/jpeg", ".jpg")
    mimetypes.add_type("image/gif", ".gif")
    mimetypes.add_type("image/webp", ".webp")


    # 自定义静态资源处理
    async def serve_custom_static(path: str, request: Request):
        if path not in custom_static_assets:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = Path(custom_static_assets[path])
        try:
            return await create_no_cache_response(file_path, request)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"处理自定义静态资源时出错: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/")
    async def index(request: Request):
        try:
            index_path = Path(STATIC_FOLDER) / "index.html"
            if not index_path.exists():
                return HTMLResponse(content=ERROR_MESSAGE.replace("TARGET_DIR", STATIC_FOLDER))

            return await create_no_cache_response(index_path, request)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error serving index: {e}")
            return HTMLResponse(content=ERROR_MESSAGE.replace("TARGET_DIR", STATIC_FOLDER))

    @app.middleware("http")
    async def spa_middleware(request: Request, call_next):
        path = request.url.path
        # 如果请求路径在自定义静态资源列表中，则返回自定义静态资源
        if path in custom_static_assets:
            return await serve_custom_static(path, request)

        skip_paths = [route.path for route in app.routes] # type: ignore

        # 如果路径在跳过路径列表中，则直接返回
        if any(path == skip_path for skip_path in skip_paths):
            return await call_next(request)

        skip_paths.remove("/")

        # 如果路径以 backend-api 开头，交由内置路由处理
        if any(path.startswith(skip_path) for skip_path in skip_paths):
            return await call_next(request)

        file_path = Path(STATIC_FOLDER) / path.lstrip('/')
        # 检查路径穿越
        if not file_path.resolve().is_relative_to(Path(STATIC_FOLDER).resolve()):
            raise HTTPException(status_code=404, detail="Access denied")

        # 如果文件存在，返回文件并禁止缓存
        if file_path.is_file():
            try:
                return await create_no_cache_response(file_path, request)
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"处理静态文件时出错: {e}")
                return FileResponse(file_path)  # 退回到普通文件响应

        fallback_path = Path(STATIC_FOLDER) / "index.html"
        # 否则返回 index.html（SPA 路由）
        if fallback_path.is_file():
            try:
                return await create_no_cache_response(fallback_path, request)
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"处理index.html时出错: {e}")
                return FileResponse(fallback_path)  # 退回到普通文件响应
        else:
            return PlainTextResponse(status_code=404, content="route not found")

    return app


class WebServer:
    app: FastAPI
    web_api_app: Quart
    listen_host: str
    listen_port: int
    container: DependencyContainer

    def __init__(self, container: DependencyContainer):
        self.app = create_app(container)
        self.web_api_app = create_web_api_app(container)
        self.server_task = None
        self.shutdown_event = asyncio.Event()
        self.container = container
        container.register(
            AuthService,
            FileBasedAuthService(
                password_file=Path(container.resolve(GlobalConfig).web.password_file),
                secret_key=container.resolve(GlobalConfig).web.secret_key,
            ),
        )
        self.config = container.resolve(GlobalConfig)

        # 配置 hypercorn
        from hypercorn.logging import Logger

        self.hypercorn_config = Config()
        self.hypercorn_config._log = Logger(self.hypercorn_config)

        # 创建自定义的日志包装器，添加 URL 过滤
        class FilteredLoggerWrapper(HypercornLoggerWrapper):
            def info(self, message, *args, **kwargs):
                # 过滤掉不需要记录的URL请求日志
                ignored_paths = [
                    '/backend-api/api/system/status',  # 添加需要过滤的URL路径
                    '/favicon.ico',
                ]
                for path in ignored_paths:
                    if path in str(args):
                        return
                super().info(message, *args, **kwargs)

        # 使用新的过滤日志包装器
        self.hypercorn_config._log.access_logger = FilteredLoggerWrapper(logger) # type: ignore
        self.hypercorn_config._log.error_logger = HypercornLoggerWrapper(logger) # type: ignore

        # 挂载 Web API 应用
        self.mount_app("/backend-api", self.web_api_app)

    def mount_app(self, prefix: str, app):
        """挂载子应用到指定路径前缀"""
        self.app.mount(prefix, app)

    def _check_port_available(self, host: str, port: int) -> bool:
        """检查端口是否可用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return True
            except socket.error:
                return False

    async def start(self):
        """启动Web服务器"""

        # 确定最终使用的host和port
        if self.container.has("cli_args"):
            cli_args = self.container.resolve("cli_args")
            self.listen_host = cli_args.host or self.config.web.host
            self.listen_port = cli_args.port or self.config.web.port
        else:
            self.listen_host = self.config.web.host
            self.listen_port = self.config.web.port

        self.hypercorn_config.bind = [f"{self.listen_host}:{self.listen_port}"]

        # 检查端口是否被占用
        if not self._check_port_available(self.listen_host, self.listen_port):
            error_msg = f"端口 {self.listen_port} 已被占用，无法启动服务器，请修改端口或关闭其他占用端口的程序。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.server_task = asyncio.create_task(serve(self.app, self.hypercorn_config, shutdown_trigger=self.shutdown_event.wait)) # type: ignore
        logger.info(f"监听地址：http://{self.listen_host}:{self.listen_port}/")
                
        # 检查WebUI是否存在，如果不存在则尝试自动安装
        self._check_and_install_webui()
        
    async def stop(self):
        """停止Web服务器"""
        self.shutdown_event.set()

        if self.server_task:
            try:
                await asyncio.wait_for(self.server_task, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Server shutdown timed out after 3 seconds.")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")

    def add_static_assets(self, url_path: str, local_path: str):
        """添加自定义静态资源"""
        if not os.path.exists(local_path):
            logger.warning(f"Static asset path does not exist: {local_path}")
            return

        custom_static_assets[url_path] = local_path

    def _check_and_install_webui(self):
        """检查WebUI是否存在，如果不存在则尝试自动安装"""
        index_path = Path(STATIC_FOLDER) / "index.html"
        if not index_path.exists():
            logger.info("检测到WebUI不存在，将在服务器启动后自动安装...")
            # 创建异步任务，但不等待完成
            self._webui_install_task = asyncio.create_task(self._install_webui())
        
    async def _install_webui(self):
        """安装WebUI的异步任务"""
        try:
            logger.info("开始自动安装WebUI...")
            success, message = await install_webui(Path(STATIC_FOLDER))
            
            if success:
                logger.info(message)
                logger.info(f"WebUI已安装到 {STATIC_FOLDER}，请刷新浏览器")
            else:
                logger.error(message)
                logger.error("WebUI自动安装失败，请手动下载并安装")
        except Exception as e:
            logger.error(f"WebUI安装过程出错: {e}")
