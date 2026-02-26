import uvicorn

from Neo4j.neo4j_api import neo4j_app

uvicorn.run(neo4j_app, host="0.0.0.0", port=8002)