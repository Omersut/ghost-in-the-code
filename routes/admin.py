from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yaml
from pathlib import Path
import git

router = APIRouter(prefix="/admin", tags=["admin"])

# Global service variables
project_assistant = None
repo_manager = None

# Function to setup services
def setup_services(pa, rm):
    global project_assistant, repo_manager
    project_assistant = pa
    repo_manager = rm
    print(f"Admin services setup - PA: {project_assistant}, RM: {repo_manager}")

class RepoConfig(BaseModel):
    name: str
    url: str
    branch: Optional[str] = "main"
    description: Optional[str] = ""

class RepoAction(BaseModel):
    repo_id: str
    action: str  # "clone", "pull", "process", "delete"

@router.get("/repos")
async def list_repos():
    """List all repositories"""
    try:
        with open("repos.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("repositories", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/repos")
async def add_repo(repo: RepoConfig):
    """Add new repository"""
    try:
        # Read current configuration
        with open("repos.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {"repositories": []}
        
        # Check if repo already exists
        if any(r["name"] == repo.name for r in config["repositories"]):
            raise HTTPException(status_code=400, detail="Repository already exists")
        
        # Add new repository
        config["repositories"].append({
            "name": repo.name,
            "url": repo.url,
            "branch": repo.branch,
            "description": repo.description
        })
        
        # Save configuration
        with open("repos.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f)
            
        return {"message": "Repository added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/repos/action")
async def repo_action(action: RepoAction):
    """Execute action on repository"""
    try:
        print(f"Executing action {action.action} on repo {action.repo_id}")
        
        if action.action == "clone":
            # Check repository info
            with open("repos.yaml", "r") as f:
                config = yaml.safe_load(f)
                repo = next((r for r in config["repositories"] if r["name"] == action.repo_id), None)
                
            if not repo:
                raise HTTPException(status_code=404, detail=f"Repository {action.repo_id} not found")
                
            # Clone repository
            repo_path = repo_manager.clone_or_pull_repo(action.repo_id)
            return {"message": "Repository cloned successfully"}
            
        elif action.action == "pull":
            # Update repository
            repo_manager.pull_repo(action.repo_id)
            return {"message": "Repository updated successfully"}
            
        elif action.action == "process":
            # Process repository
            repo_path = repo_manager.base_path / action.repo_id
            project_assistant.process_repo(action.repo_id, repo_path)
            return {"message": "Repository processed successfully"}
            
        elif action.action == "delete":
            # Delete repository
            repo_manager.delete_repo(action.repo_id)
            return {"message": "Repository deleted successfully"}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/repos/status")
async def get_repo_status():
    """Check repository status"""
    try:
        if not Path("repos.yaml").exists():
            with open("repos.yaml", "w", encoding="utf-8") as f:
                yaml.dump({"repositories": []}, f)
        
        with open("repos.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {"repositories": []}
            
        repos = []
        for repo in config.get("repositories", []):
            repo_path = Path(f"repos/{repo['name']}")
            
            try:
                collection_exists = repo["name"] in project_assistant.collections
            except Exception as e:
                collection_exists = False
            
            status = {
                "name": repo["name"],
                "url": repo["url"],
                "description": repo.get("description", ""),
                "is_cloned": repo_path.exists(),
                "is_processed": collection_exists,
                "last_updated": None
            }
            
            if repo_path.exists():
                try:
                    git_repo = git.Repo(repo_path)
                    status["last_updated"] = git_repo.head.commit.committed_datetime.isoformat()
                except Exception as e:
                    print(f"Error getting git info: {str(e)}")
                    
            repos.append(status)
            
        return repos
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 