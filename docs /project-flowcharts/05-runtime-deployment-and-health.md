# Runtime, Deployment and Health Flow

```mermaid
flowchart TD
    A["Start project"] --> B{"Startup method"}
    B -->|start.sh or run.sh| C["Activate Python virtual environment<br/>Install dependencies when missing"]
    B -->|Docker Compose| D["Start four containers"]

    C --> E["Check Docker Desktop"]
    E --> F["Start Qdrant on 6333<br/>and Elasticsearch on 9200"]
    F --> G["Start FastAPI on 8000"]
    F --> H["Start Next.js on 3000"]

    D --> I["Qdrant container<br/>Persistent ./qdrant_storage volume<br/>HTTP 6333 and gRPC 6334"]
    D --> J["Elasticsearch 8.17 single node<br/>Security disabled<br/>512 MB JVM heap"]
    D --> K["FastAPI backend container"]
    D --> L["Next.js frontend container"]
    I --> K
    J --> K
    K --> L

    G --> M["FastAPI lifespan initializes<br/>one MultiAgentWorkflow and MemorySaver"]
    K --> M
    M --> N["Register chat, health and evaluation routers<br/>CORS currently allows every origin"]

    N --> O["GET /api/health"]
    O --> P["Check workflow object and compiled graph"]
    P --> Q["Connect to Qdrant collection<br/>Read points_count"]
    Q --> R{"Workflow healthy and Qdrant has points?"}
    R -->|Yes| S["status=healthy"]
    R -->|No| T["status=degraded"]

    N --> U["GET /api/health/ready<br/>Check workflow initialization"]
    N --> V["GET /api/health/live<br/>Return alive=true"]
```

Deployment caveats in the current code:

- Docker Compose sets `QDRANT_HOST` and `QDRANT_PORT`, but the backend reads `QDRANT_URL`.
- Docker Compose sets `ELASTICSEARCH_URL`, but the backend reads `ES_URL`.
- Local Elasticsearch is ignored unless `ES_ENABLED=true`.
- The API advertises `/api/chat/stream`, but that route is not implemented.
- The chat route supplies `node_timings`, but the backend `ChatResponse` model does not declare that field.
- The frontend Stop control stops visual playback, not an active backend request.

Source files:

- `start.sh`
- `run.sh`
- `docker-compose.yml`
- `Dockerfile`
- `render.yaml`
- `src/api/main.py`
- `src/api/route/health.py`
- `src/config/clients.py`

