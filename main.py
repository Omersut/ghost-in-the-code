from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from utils import initialize_system, RepoManager
import asyncio
from routes.admin import router, setup_services  # setup_services'i import et

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Önce servisleri oluştur
project_assistant, repo_manager = initialize_system()

# Servisleri admin router'a ayarla
setup_services(project_assistant, repo_manager)

# Router'ı app'e ekle
app.include_router(router)

class Query(BaseModel):
    question: str
    query_type: str = "all"  # "all", "repo", "files"
    project_id: Optional[str] = None
    files: Optional[List[str]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/repos/list")
async def list_repos():
    return project_assistant.list_repos()

@app.post("/query")
async def query(query: Query):
    try:
        # StreamingResponse ile sarmalayalım
        return StreamingResponse(
            generate_response(query),
            media_type='text/event-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_response(query: Query):
    """Yanıtları stream olarak gönder"""
    try:
        if query.query_type == "all":
            async for chunk in project_assistant.query_all(query.question):
                yield f"data: {chunk}\n\n"
                
        elif query.query_type == "repo":
            async for chunk in project_assistant.query_repo(query.project_id, query.question):
                yield f"data: {chunk}\n\n"
                
        elif query.query_type == "files":
            async for chunk in project_assistant.query_files(query.project_id, query.files, query.question):
                yield f"data: {chunk}\n\n"
                
        else:
            yield f"data: Error: Invalid query type\n\n"
            
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

@app.post("/project/{project_id}/query")
async def query_project(project_id: str, query: Query):
    return await project_assistant.query(
        project_id=project_id,
        question=query.question
    )

@app.get("/project/{project_id}/metrics")
async def get_project_metrics(project_id: str):
    return project_assistant.get_metrics(project_id)

@app.get("/project/{project_id}/history")
async def get_chat_history(project_id: str):
    return project_assistant.get_chat_history(project_id)

@app.get("/project/{project_id}/structure")
async def get_project_structure(project_id: str):
    return project_assistant.get_project_structure(project_id)

@app.get("/projects")
async def list_projects():
    try:
        projects = project_assistant.list_projects()
        print(f"Available projects: {projects}")  # Debug için
        return projects
    except Exception as e:
        print(f"Error in list_projects: {e}")
        return []

@app.get("/project/{project_id}/file/{file_path:path}")
async def get_file_content(project_id: str, file_path: str):
    return project_assistant.get_file_content(project_id, file_path)

@app.get("/project/{project_id}/file/history/{file_path:path}")
async def get_file_history(project_id: str, file_path: str):
    return project_assistant.get_file_history(project_id, file_path)

@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html") 