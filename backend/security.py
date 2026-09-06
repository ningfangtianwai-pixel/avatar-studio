"""Browser-origin and streamed-body guards for a loopback-only desktop service."""
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from backend.config import ALLOWED_ORIGINS, MAX_REQUEST_BYTES


class LocalRequestGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        origin = headers.get(b"origin", b"").decode("latin1")
        method = scope["method"]
        reason = None
        code = 403
        if origin and origin not in ALLOWED_ORIGINS:
            reason = "只允许工作台页面访问本机接口。"
        if method not in {"GET", "HEAD", "OPTIONS"}:
            if headers.get(b"x-avatar-studio") != b"1":
                reason = "请求缺少工作台标识，请刷新页面后重试。"
            try:
                length = int(headers.get(b"content-length", b"0"))
                if length < 0 or length > MAX_REQUEST_BYTES:
                    reason, code = "上传内容超过 256 MB 限制。", 413
            except ValueError:
                reason, code = "无效的请求长度。", 400
        if reason:
            return await JSONResponse({"detail": reason}, status_code=code)(scope, receive, send)
        consumed = 0
        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > MAX_REQUEST_BYTES:
                raise HTTPException(413, "上传内容超过 256 MB 限制。")
            return message
        await self.app(scope, limited_receive, send)
