# MCP 服务器管理 API

MCP（Model Context Protocol）是一种用于大型语言模型的通信协议。本 API 提供了管理 MCP 服务器的功能，包括创建、更新、删除、启动和停止服务器，以及获取服务器提供的工具列表。

## 数据模型

### MCP 服务器（MCPServer）

```json
{
  "id": "claude-mcp",
  "description": "Claude MCP 服务器",
  "command": "python",
  "args": "-m claude_cli.mcp",
  "connection_type": "stdio",
  "status": "stopped",
  "error_message": null,
  "created_at": "2023-04-01T12:00:00",
  "last_used_at": null
}
```

### 连接类型（ConnectionType）

- `stdio`: 标准输入输出连接
- `sse`: 服务器发送事件连接

### 服务器状态（ServerStatus）

- `running`: 运行中
- `stopped`: 已停止
- `error`: 错误状态

## API 端点

### 获取服务器列表

获取 MCP 服务器列表，支持分页和过滤。

**请求**：
```
POST /mcp/servers
```

**请求体**：
```json
{
  "page": 1,
  "page_size": 20,
  "connection_type": null,
  "status": null,
  "query": null
}
```

**响应**：
```json
{
  "items": [
    {
      "id": "claude-mcp",
      "description": "Claude MCP 服务器",
      "command": "python",
      "args": "-m claude_cli.mcp",
      "connection_type": "stdio",
      "status": "stopped",
      "error_message": null,
      "created_at": "2023-04-01T12:00:00",
      "last_used_at": null
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

### 获取统计信息

获取 MCP 服务器相关的统计信息。

**请求**：
```
GET /mcp/statistics
```

**响应**：
```json
{
  "total_servers": 3,
  "stdio_servers": 2,
  "sse_servers": 1,
  "running_servers": 1,
  "stopped_servers": 2
}
```

### 获取服务器详情

获取特定 MCP 服务器的详细信息。

**请求**：
```
GET /mcp/servers/{server_id}
```

**响应**：
```json
{
  "id": "claude-mcp",
  "description": "Claude MCP 服务器",
  "command": "python",
  "args": "-m claude_cli.mcp",
  "connection_type": "stdio",
  "status": "stopped",
  "error_message": null,
  "created_at": "2023-04-01T12:00:00",
  "last_used_at": null
}
```

### 获取服务器工具列表

获取 MCP 服务器提供的工具列表。

**请求**：
```
GET /mcp/servers/{server_id}/tools
```

**响应**：
```json
[
  {
    "name": "search_web",
    "description": "搜索网络获取信息",
    "parameters": {
      "query": {
        "type": "string",
        "description": "搜索查询"
      }
    }
  },
  {
    "name": "analyze_image",
    "description": "分析图像内容",
    "parameters": {
      "image_url": {
        "type": "string",
        "description": "图像URL"
      },
      "detect_faces": {
        "type": "boolean",
        "description": "是否检测人脸"
      }
    }
  }
]
```

### 检查服务器 ID 是否可用

检查给定的服务器 ID 是否已存在。

**请求**：
```
GET /mcp/servers/check/{server_id}
```

**响应**：
返回布尔值，`true` 表示 ID 可用，`false` 表示 ID 已存在。

### 创建服务器

创建新的 MCP 服务器。

**请求**：
```
POST /mcp/servers/create
```

**请求体**：
```json
{
  "id": "claude-mcp",
  "description": "Claude MCP 服务器",
  "command": "python",
  "args": "-m claude_cli.mcp",
  "connection_type": "stdio"
}
```

**响应**：
返回创建的服务器详情。

### 更新服务器

更新现有 MCP 服务器的配置。

**请求**：
```
PUT /mcp/servers/{server_id}
```

**请求体**：
```json
{
  "description": "Updated description",
  "command": "python3",
  "args": "-m claude_cli.mcp --verbose",
  "connection_type": "stdio"
}
```

**响应**：
返回更新后的服务器详情。

### 删除服务器

删除 MCP 服务器。

**请求**：
```
DELETE /mcp/servers/{server_id}
```

**响应**：
```json
{
  "message": "服务器已成功删除"
}
```

### 启动服务器

启动 MCP 服务器。

**请求**：
```
POST /mcp/servers/{server_id}/start
```

**响应**：
```json
{
  "message": "服务器已启动"
}
```

### 停止服务器

停止 MCP 服务器。

**请求**：
```
POST /mcp/servers/{server_id}/stop
```

**响应**：
```json
{
  "message": "服务器已停止"
}
```

## 使用示例

### 创建并启动 MCP 服务器

1. 创建服务器：
```
POST /mcp/servers/create
```
```json
{
  "id": "claude-mcp",
  "description": "Claude MCP 服务器",
  "command": "python",
  "args": "-m claude_cli.mcp",
  "connection_type": "stdio"
}
```

2. 启动服务器：
```
POST /mcp/servers/claude-mcp/start
```

3. 获取服务器提供的工具：
```
GET /mcp/servers/claude-mcp/tools
```

## 注意事项

1. 在更新服务器配置前，必须先停止正在运行的服务器。
2. 服务器 ID 必须是唯一的，可以使用 `/mcp/servers/check/{server_id}` 接口检查 ID 是否可用。
3. 创建服务器后，服务器状态默认为 `stopped`，需要手动启动。 