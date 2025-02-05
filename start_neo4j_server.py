import uvicorn

from Neo4j.neo4j_api import neo4j_app

#TODO In teoria morirà, se davvero dockerizziamo

uvicorn.run(neo4j_app, port=8002)