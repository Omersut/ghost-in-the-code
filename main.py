from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from utils import initialize_system, RepoManager
import asyncio
from routes.admin import router, setup_services  # setup_services'i import et
import git
from pathlib import Path
import markdown2  # pip install markdown2
import os

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

@app.post("/select-files")
async def select_files(request: Request):
    data = await request.json()
    repo_id = data.get("repo_id")
    selected_files = data.get("files", [])
    
    try:
        result = project_assistant.set_context(
            context_type="files",
            repo=repo_id,
            files=selected_files
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-files/{repo_id}")
async def get_available_files(repo_id: str):
    try:
        # Önce repo context'ini ayarla
        project_assistant.set_context(
            context_type="repo",
            repo=repo_id
        )
        # Dosya listesini al
        context_info = project_assistant.get_context_info()
        return context_info["available_files"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repo/{repo_id:path}/details")
async def repo_details(repo_id: str):
    # URL'deki özel karakterleri düzelt
    repo_id = repo_id.replace("%20", " ").replace("-", " ")
    return FileResponse("static/repo-details.html")

@app.get("/api/repo/{repo_id}/commits")
async def get_repo_commits(repo_id: str):
    try:
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        commits = []
        for commit in repo.iter_commits('--all', max_count=20):  # Son 20 commit
            commits.append({
                'hash': commit.hexsha,
                'message': commit.message,
                'author': commit.author.name,
                'date': commit.committed_datetime.isoformat(),
                'stats': {
                    'files': len(commit.stats.files),
                    'insertions': commit.stats.total['insertions'],
                    'deletions': commit.stats.total['deletions']
                }
            })
        
        return commits
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id}/file-history/{file_path:path}")
async def get_file_history(repo_id: str, file_path: str):
    try:
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        commits = []
        for commit in repo.iter_commits('--all', paths=file_path):
            commits.append({
                'hash': commit.hexsha,
                'message': commit.message,
                'author': commit.author.name,
                'date': commit.committed_datetime.isoformat()
            })
        
        return commits
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id}/branches")
async def get_repo_branches(repo_id: str):
    try:
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        branches = []
        for branch in repo.heads:
            branches.append({
                'name': branch.name,
                'commit': {
                    'hash': branch.commit.hexsha,
                    'message': branch.commit.message,
                    'author': branch.commit.author.name,
                    'date': branch.commit.committed_datetime.isoformat()
                },
                'is_active': branch.name == repo.active_branch.name
            })
        
        return branches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id}/file-content/{file_path:path}")
async def get_file_content(repo_id: str, file_path: str, commit: Optional[str] = None):
    try:
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        try:
            if commit:
                # Belirli bir commit'teki içeriği al
                content = repo.git.show(f"{commit}:{file_path}")
            else:
                # En son versiyondaki içeriği al
                file_path_obj = repo_path / file_path
                content = file_path_obj.read_text(errors='replace')
            
            # Markdown dosyaları için HTML'e çevir
            if file_path.lower().endswith('.md'):
                content = markdown2.markdown(content, extras=['fenced-code-blocks', 'tables'])
                return {"content": content, "is_markdown": True}
            
            return {"content": content, "is_markdown": False}
            
        except UnicodeDecodeError:
            # Binary dosyalar için
            return {
                "content": "Binary file not shown",
                "is_binary": True
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id}/file-diff/{file_path:path}")
async def get_file_diff(repo_id: str, file_path: str, commit1: str, commit2: str):
    try:
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        diff = repo.git.diff(commit1, commit2, "--", file_path)
        return {"diff": diff}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id:path}/structure")
async def get_repo_structure(repo_id: str):
    try:
        # URL'deki özel karakterleri düzelt
        repo_id = repo_id.replace("%20", " ").replace("-", " ")
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        def build_tree(path):
            tree = []
            for item in path.iterdir():
                if item.name.startswith('.'):
                    continue
                    
                node = {
                    "name": item.name,
                    "path": str(item.relative_to(repo_path)),
                    "type": "file" if item.is_file() else "folder"
                }
                
                if item.is_dir():
                    node["children"] = build_tree(item)
                    
                tree.append(node)
            return sorted(tree, key=lambda x: (x["type"] == "file", x["name"]))
            
        return build_tree(repo_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id:path}/stats")
async def get_repo_stats(repo_id: str):
    try:
        repo_id = repo_id.replace(" ", "-").replace("%20", "-")
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        # Repo istatistikleri
        stats = {
            'total_commits': sum(1 for _ in repo.iter_commits()),
            'total_branches': len(repo.heads),
            'active_branch': repo.active_branch.name,
            'contributors': [],
            'file_types': {},
            'last_activity': None,
            'total_files': 0,
            'total_size': 0
        }
        
        # Contributor istatistikleri
        for commit in repo.iter_commits():
            contributor = {
                'name': commit.author.name,
                'email': commit.author.email,
                'date': commit.authored_datetime.isoformat()
            }
            if contributor not in stats['contributors']:
                stats['contributors'].append(contributor)
            
            if not stats['last_activity'] or commit.authored_datetime > stats['last_activity']:
                stats['last_activity'] = commit.authored_datetime
        
        # Dosya türü istatistikleri
        for root, _, files in os.walk(repo_path):
            if '.git' in root:
                continue
            for file in files:
                stats['total_files'] += 1
                file_path = Path(root) / file
                stats['total_size'] += file_path.stat().st_size
                ext = file_path.suffix.lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id}/recent-activity")
async def get_recent_activity(repo_id: str, limit: int = 10):
    try:
        repo_path = repo_manager.base_path / repo_id
        repo = git.Repo(repo_path)
        
        activities = []
        for commit in repo.iter_commits(max_count=limit):
            # Her commit için değişen dosyaları al
            changed_files = []
            for item in commit.stats.files:
                changed_files.append({
                    'path': item,
                    'changes': commit.stats.files[item]
                })
            
            activities.append({
                'type': 'commit',
                'hash': commit.hexsha,
                'short_hash': commit.hexsha[:7],
                'message': commit.message,
                'author': {
                    'name': commit.author.name,
                    'email': commit.author.email
                },
                'date': commit.authored_datetime.isoformat(),
                'changed_files': changed_files
            })
        
        return activities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id:path}/file-preview/{file_path:path}")
async def get_file_preview(repo_id: str, file_path: str):
    try:
        repo_id = repo_id.replace("%20", " ").replace("-", " ")
        repo_path = repo_manager.base_path / repo_id
        file_path_full = repo_path / file_path
        
        # Binary dosya kontrolü
        try:
            with open(file_path_full, 'rb') as f:
                content = f.read(1024)  # İlk 1KB'ı oku
                if b'\x00' in content:
                    return {"preview": "Binary file", "is_binary": True}
        except:
            return {"preview": "Error reading file", "error": True}
        
        # Text dosyası ise
        try:
            with open(file_path_full, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                preview = ''.join(lines[:10])  # İlk 10 satır
                total_lines = len(lines)
                
                return {
                    "preview": preview,
                    "total_lines": total_lines,
                    "is_binary": False,
                    "size": file_path_full.stat().st_size
                }
        except UnicodeDecodeError:
            return {"preview": "Binary file", "is_binary": True}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 