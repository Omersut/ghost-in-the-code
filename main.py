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
import shutil

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
    try:
        repos = []
        for repo in project_assistant.list_repos():
            repos.append({
                "name": repo["name"],
                "local_path": repo["local_path"],
                "is_processed": repo.get("is_processed", False)
            })
        return repos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        context = data.get("context", {})
        
        print(f"Processing query: {question}")
        print(f"Context: {context}")
        
        # Stream yanıtı döndür
        response = await project_assistant.query(question, context)
        
        # Response zaten bir StreamingResponse ise direkt döndür
        if isinstance(response, StreamingResponse):
            return response
        
        # String yanıtları stream formatına çevir
        if isinstance(response, (str, bytes)):
            return StreamingResponse(
                iter([f"data: {response}\n\n"]),
                media_type='text/event-stream'
            )
        
        # Beklenmeyen yanıt tipi
        return StreamingResponse(
            iter([f"data: Error: Unexpected response type\n\n"]),
            media_type='text/event-stream'
        )

    except Exception as e:
        print(f"Error in query endpoint: {e}")
        return StreamingResponse(
            iter([f"data: Error: {str(e)}\n\n"]),
            media_type='text/event-stream'
        )

async def generate_response(query: Query):
    """Yanıtları stream olarak gönder"""
    try:
        print(f"Generating response for query type: {query.query_type}")
        if query.query_type == "all":
            async for chunk in project_assistant.query_all(query.question):
                yield f"data: {chunk}\n\n"
                
        elif query.query_type == "repo":
            print(f"Querying repo: {query.project_id}")
            async for chunk in project_assistant.query_repo(query.project_id, query.question):
                yield f"data: {chunk}\n\n"
                
        elif query.query_type == "files":
            print(f"Querying files in repo {query.project_id}: {query.files}")
            async for chunk in project_assistant.query_files(query.project_id, query.files, query.question):
                yield f"data: {chunk}\n\n"
                
        else:
            yield f"data: Error: Invalid query type\n\n"
            
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
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

@app.get("/project/{repo_id}/structure")
async def get_project_structure(repo_id: str):
    """Get project file structure"""
    try:
        structure = project_assistant.get_repo_structure(repo_id)
        return structure
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/repo/{repo_id:path}/details/stats")
async def get_repo_stats(repo_id: str):
    try:
        # Repo bilgilerini config'den al
        repo = next((r for r in project_assistant.config["repositories"] if r["name"] == repo_id), None)
        if not repo:
            raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found in config")

        # Local path'i kullan
        repo_path = Path(repo["local_path"])
        
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail=f"Repository not found: {repo_path}")
            
        if not repo_path.is_dir():
            raise HTTPException(status_code=400, detail="Not a directory")

        try:
            repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            raise HTTPException(status_code=400, detail="Not a valid git repository")
        
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
            if not any(c['email'] == contributor['email'] for c in stats['contributors']):
                stats['contributors'].append(contributor)
            
            if not stats['last_activity'] or commit.authored_datetime > stats['last_activity']:
                stats['last_activity'] = commit.authored_datetime
        
        # Dosya istatistikleri
        for root, _, files in os.walk(repo_path):
            if '.git' in root.split(os.sep):
                continue
            for file in files:
                stats['total_files'] += 1
                file_path = Path(root) / file
                try:
                    stats['total_size'] += file_path.stat().st_size
                except:
                    continue
                ext = file_path.suffix.lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
        
        return stats
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error getting repo stats: {str(e)}")

@app.get("/api/repo/{repo_id:path}/details/recent-activity")
async def get_repo_recent_activity(repo_id: str, limit: int = 20):
    try:
        # Repo bilgilerini config'den al
        repo = next((r for r in project_assistant.config["repositories"] if r["name"] == repo_id), None)
        if not repo:
            raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found in config")

        # Local path'i kullan
        repo_path = Path(repo["local_path"])
        
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail=f"Repository not found: {repo_path}")
            
        if not repo_path.is_dir():
            raise HTTPException(status_code=400, detail="Not a directory")

        try:
            repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            raise HTTPException(status_code=400, detail="Not a valid git repository")
        
        activities = []
        for commit in repo.iter_commits(max_count=limit):
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
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error getting repo activity: {str(e)}")

@app.get("/repo/{repo_id:path}/details")
async def repo_details_page(repo_id: str):
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

@app.get("/api/repo/{repo_id:path}/file-content/{file_path:path}")
async def get_file_content(repo_id: str, file_path: str):
    try:
        # URL decode ve path düzeltme
        repo_id = repo_id.replace("%20", " ")
        repo_path = repo_manager.base_path / repo_id.replace(" ", "-")
        
        # Dosya yolunu düzelt
        file_path = file_path.replace("\\", "/")
        file_path_full = repo_path / file_path
        
        if not file_path_full.exists():
            return {"error": True, "message": f"File not found: {file_path}"}
            
        # Markdown dosyası kontrolü
        if file_path.lower().endswith('.md'):
            with open(file_path_full, 'r', encoding='utf-8') as f:
                content = f.read()
                html_content = markdown2.markdown(content)
                return {"content": html_content, "is_markdown": True}
        
        # Binary dosya kontrolü
        try:
            with open(file_path_full, 'rb') as f:
                content = f.read(1024)  # İlk 1KB kontrol
                if b'\x00' in content:
                    return {"content": "Binary file", "is_binary": True}
                    
            # Text dosyası okuma
            with open(file_path_full, 'r', encoding='utf-8') as f:
                content = f.read()
                return {
                    "content": content,
                    "is_binary": False,
                    "is_markdown": False
                }
                
        except UnicodeDecodeError:
            return {"content": "Binary file", "is_binary": True}
            
    except Exception as e:
        print(f"Error reading file: {str(e)}")  # Debug için
        return {"error": True, "message": str(e)}

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
        # URL decode ve path düzeltme
        repo_id = repo_id.replace("%20", " ")  # URL decode
        repo_path = repo_manager.base_path / repo_id  # Direkt repo_id kullan
        
        if not repo_path.exists():
            # Alternatif yolları dene
            alt_path = repo_manager.base_path / repo_id.replace(" ", "-")
            if alt_path.exists():
                repo_path = alt_path
            else:
                raise HTTPException(status_code=404, detail="Repository not found")
        
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

@app.post("/admin/repos/add-local")
async def add_local_repo(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        local_path = data.get("localPath")

        # Windows path'lerini düzelt
        local_path = local_path.replace("\\", "/")

        # Repo bilgilerini repos.yaml'a ekle
        repo_config = {
            "name": name,
            "local_path": local_path
        }
        
        # repos.yaml'ı güncelle - path parametresini de gönder
        project_assistant.add_repo(repo_config, local_path)
        
        return {"success": True, "message": "Repository added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/repos/action")
async def repo_action(request: Request):
    try:
        data = await request.json()
        repo_id = data.get("repo_id")
        action = data.get("action")
        
        if action == "process":
            # Repo'yu işle
            await project_assistant.process_repo(repo_id)
            return {"success": True, "message": "Repository processed successfully"}
            
        elif action == "delete":
            # Repo'yu sil (sadece yapılandırmadan)
            project_assistant.remove_repo(repo_id)
            return {"success": True, "message": "Repository removed successfully"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repo/{repo_id:path}/commit/{commit_hash}")
async def get_commit_details(repo_id: str, commit_hash: str):
    try:
        repo = next((r for r in project_assistant.config["repositories"] if r["name"] == repo_id), None)
        if not repo:
            raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found in config")

        repo_path = Path(repo["local_path"])
        git_repo = git.Repo(repo_path)
        
        try:
            commit = git_repo.commit(commit_hash)
        except git.exc.BadName:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        changed_files = []
        for parent in commit.parents:
            diff = parent.diff(commit)
            for d in diff:
                changed_files.append({
                    'path': d.a_path,
                    'diff': git_repo.git.diff(parent.hexsha, commit.hexsha, d.a_path)
                })
        
        return {
            'hash': commit.hexsha,
            'short_hash': commit.hexsha[:7],
            'message': commit.message,
            'author': {
                'name': commit.author.name,
                'email': commit.author.email
            },
            'date': commit.authored_datetime.isoformat(),
            'changed_files': changed_files
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e)) 