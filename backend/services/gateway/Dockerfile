# Build context is the backend root (/opt/haru) so the shared protocol
# package, which the deploy script copies to ./protocol, is reachable.
FROM python:3.13-slim

RUN useradd --system --uid 10001 haru
WORKDIR /app

# Silero VAD v5 (MIT), pinned by tag + checksum. Runs on CPU in the gateway.
ADD --checksum=sha256:2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f \
    https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx \
    /opt/silero/silero_vad.onnx
RUN chmod 644 /opt/silero/silero_vad.onnx

# GitHub's official MCP server (stdio), pinned by version + checksum. Run by the
# gateway as a subprocess, read-only and in lockdown mode (see tools/mcp.py).
ADD --checksum=sha256:95843162759da2c31dde082dd145be35db82164594796c294414b69790c2290e \
    https://github.com/github/github-mcp-server/releases/download/v1.12.2/github-mcp-server_Linux_x86_64.tar.gz \
    /tmp/github-mcp-server.tar.gz
RUN tar -xzf /tmp/github-mcp-server.tar.gz -C /usr/local/bin github-mcp-server \
 && chmod 755 /usr/local/bin/github-mcp-server && rm /tmp/github-mcp-server.tar.gz
COPY protocol /src/protocol
COPY services/gateway /src/gateway
RUN pip install --no-cache-dir /src/protocol /src/gateway

USER haru
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "haru_gateway"]
