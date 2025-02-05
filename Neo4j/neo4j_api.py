from Neo4j.neo4j_console import Neo4jClient

from fastapi import FastAPI

neo4j_app = FastAPI()

neo4j_server = Neo4jClient()

@neo4j_app.post("/start")
def start():
    return neo4j_server._start_console()

@neo4j_app.post("/stop")
def stop():
    return neo4j_server._stop_console()

@neo4j_app.get("/status")
def status():
    return {"running": neo4j_server.is_running()}