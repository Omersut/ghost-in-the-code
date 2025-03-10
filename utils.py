import git
from pathlib import Path
import yaml
import os
from typing import List, Optional, Dict
import httpx
import json
import chromadb
from chromadb.utils import embedding_functions
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime
from starlette.responses import FileResponse, StreamingResponse
import asyncio
import requests
import errno
import stat
import shutil
from fastapi import HTTPException
import markdown2
from functools import lru_cache
import time
import re

class RepoManager:
    def __init__(self, base_path: str = "repos"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self._active_repos = {}  # Track active repo instances

    def clone_or_pull_repo(self, repo_name: str) -> Path:
        """Repoyu klonla veya güncelle"""
        try:
            # Önce repo bağlantısını temizle
            if repo_name in self._active_repos:
                self._cleanup_repo(repo_name)

            # Repo bilgilerini al
            with open("repos.yaml", "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)
                repo_info = next((r for r in config["repositories"] if r["name"] == repo_name), None)
                
            if not repo_info:
                raise ValueError(f"Repository {repo_name} not found in config")

            repo_path = self.base_path / repo_name
            
            if repo_path.exists():
                print(f"Pulling {repo_name}...")
                repo = git.Repo(repo_path)
                # Değişiklikleri geri al
                repo.git.reset('--hard')
                repo.git.clean('-fd')
                # Pull
                repo.remotes.origin.pull()
            else:
                print(f"Cloning {repo_name}...")
                repo = git.Repo.clone_from(repo_info["url"], repo_path)
                
            # Aktif repo listesine ekle
            self._active_repos[repo_name] = repo
            return repo_path

        except Exception as e:
            print(f"Error in clone_or_pull_repo: {e}")
            # Hata durumunda temizlik
            self._cleanup_repo(repo_name)
            raise

    def _cleanup_repo(self, repo_name: str):
        """Repo instance'ını temizle"""
        try:
            if repo_name in self._active_repos:
                repo = self._active_repos[repo_name]
                try:
                    repo.git.gc()  # Git garbage collection
                    repo.close()   # Git bağlantısını kapat
                except:
                    pass
                del self._active_repos[repo_name]
        except Exception as e:
            print(f"Error cleaning up repo {repo_name}: {e}")

    def delete_repo(self, repo_name: str):
        """Repoyu sil"""
        try:
            # Önce repo bağlantısını temizle
            self._cleanup_repo(repo_name)
            
            repo_path = self.base_path / repo_name
            
            if repo_path.exists():
                def handle_remove_readonly(func, path, exc):
                    if func in (os.unlink, os.rmdir) and exc[1].errno == errno.EACCES:
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    else:
                        raise exc

                # Klasörü ve içindekileri sil
                shutil.rmtree(repo_path, onerror=handle_remove_readonly)
            
            # repos.yaml'dan kaldır
            with open("repos.yaml", "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            config["repositories"] = [
                r for r in config["repositories"] 
                if r["name"] != repo_name
            ]
            
            with open("repos.yaml", "w", encoding='utf-8') as f:
                yaml.dump(config, f)
                
        except Exception as e:
            print(f"Error in delete_repo: {e}")
            raise

    def pull_repo(self, repo_name: str):
        """Repoyu güncelle"""
        try:
            # Önce repo bağlantısını temizle
            self._cleanup_repo(repo_name)
            
            repo_path = self.base_path / repo_name
            if repo_path.exists():
                repo = git.Repo(repo_path)
                # Değişiklikleri geri al
                repo.git.reset('--hard')
                repo.git.clean('-fd')
                # Pull
                repo.remotes.origin.pull()
                # Aktif repo listesine ekle
                self._active_repos[repo_name] = repo
            else:
                raise Exception(f"Repo {repo_name} does not exist locally")
        except Exception as e:
            print(f"Error in pull_repo: {e}")
            raise

class ProjectAssistant:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=self.device
        )
        self.chat_history = {}
        self.collections = {}
        self.ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.max_retries = 5
        self.timeout = 60.0
        self._check_ollama_connection()
        self.current_context = {
            "type": "all",
            "repo": None,
            "files": []
        }
        # Config'i yükle
        self.load_config()
        # Mevcut koleksiyonları yükle
        self._load_existing_collections()
        self._structure_cache = {}  # Dosya yapısı cache'i
        self._cache_timeout = 300  # Cache timeout süresi (saniye)

    def load_config(self):
        """repos.yaml dosyasından konfigürasyonu yükle"""
        try:
            with open("repos.yaml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
                if "repositories" not in self.config:
                    self.config["repositories"] = []
        except FileNotFoundError:
            self.config = {"repositories": []}
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {"repositories": []}

    def save_config(self):
        """Konfigürasyonu repos.yaml'a kaydet"""
        try:
            with open("repos.yaml", "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True)
        except Exception as e:
            print(f"Error saving config: {e}")
            raise

    def _check_ollama_connection(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            print("Ollama connection successful")
        except Exception as e:
            print(f"Warning: Ollama connection failed - {str(e)}")
            print("Please make sure Ollama is running with: ollama serve")

    def _normalize_collection_name(self, name: str) -> str:
        """Repo ismini geçerli bir koleksiyon ismine dönüştür"""
        # Boşlukları tire ile değiştir
        normalized = name.replace(' ', '-')
        # Özel karakterleri kaldır, sadece alfanumerik ve tire bırak
        normalized = ''.join(c for c in normalized if c.isalnum() or c == '-')
        # Birden fazla tireyi tekli tireye dönüştür
        normalized = '-'.join(filter(None, normalized.split('-')))
        # Başındaki ve sonundaki tireleri kaldır
        normalized = normalized.strip('-')
        # Küçük harfe çevir
        normalized = normalized.lower()
        return normalized

    async def process_repo(self, repo_id: str, repo_path: Optional[Path] = None):
        """Process repository and create collection"""
        try:
            # Repo bilgilerini config'den al
            repo = next((r for r in self.config["repositories"] if r["name"] == repo_id), None)
            if not repo:
                raise Exception(f"Repository {repo_id} not found in config")

            # Config'deki local_path'i kullan
            if repo_path is None:
                repo_path = Path(repo["local_path"])
            
            # Process repo
            self.process_repo_content(repo_id, repo_path)
            
            # Update processed status
            repo["is_processed"] = True
            self.save_config()
            
            return {"success": True, "message": "Repository processed successfully"}
            
        except Exception as e:
            print(f"Error processing repo: {e}")
            raise

    def process_repo_content(self, repo_name: str, repo_path: Path):
        """Process repository content and create collection"""
        try:
            print(f"Processing repo: {repo_name} at {repo_path}")
            
            def update_progress(message: str):
                print(message)  # Console'a yazdır
                # Progress event'i gönder
                if hasattr(self, 'progress_callback'):
                    self.progress_callback(message)

            # İşlenmeyecek klasörler
            IGNORED_DIRS = {
                # Build ve paket klasörleri
                'node_modules',
                'bin',
                'obj',
                'dist',
                'build',
                'target',
                'packages',
                
                # Versiyon kontrol ve IDE
                '.git',
                '.vs',
                '.idea',
                '.vscode',
                '__pycache__',
                
                # Ortam klasörleri
                'venv',
                'env',
                'virtualenv',
                '.env',
                
                # Asset klasörleri
                'assets',
                'images',
                'fonts',
                'wwwroot',
                'static',
                'media',
                
                # CI/CD ve deployment
                'jenkins_home',
                '.jenkins',
                'docker',
                'kubernetes',
                'k8s',
                'helm',
                'terraform',
                'ansible',
                
                # Test ve dökümantasyon
                'test',
                'tests',
                'docs',
                'examples',
                'samples',
                
                # Geçici ve cache
                'temp',
                'tmp',
                'cache',
                'logs',
                '.sonarqube',
                'coverage',
                '.nyc_output',
                
                # Build output
                'Debug',
                'Release',
                'x64',
                'x86',
                'net4*',
                'netcoreapp*',
                'publish'
            }

            # İşlenmeyecek dosya uzantıları
            IGNORED_EXTENSIONS = {
                # Binary dosyalar
                '.exe', '.dll', '.pdb', '.so', '.dylib',
                '.pyc', '.pyo', '.pyd',
                
                # Media dosyaları
                '.jpg', '.jpeg', '.png', '.gif', '.ico', '.svg',
                '.mp3', '.mp4', '.wav', '.avi', '.mov',
                '.pdf', '.doc', '.docx', '.xls', '.xlsx',
                
                # Sıkıştırılmış dosyalar
                '.zip', '.rar', '.7z', '.tar', '.gz', '.tgz',
                
                # Minified ve generated
                '.min.js', '.min.css',
                '.bundle.js', '.bundle.css',
                '.generated.cs', '.designer.cs',
                '.g.cs', '.g.i.cs',
                
                # Source maps ve debug
                '.map', '.pdb', '.cache',
                
                # Log ve data
                '.log', '.log.*',
                '.sqlite', '.db', '.mdf', '.ldf',
                
                # Config ve secret
                '.env',
                '.env.*',
                'appsettings.*.json',
                '*.pfx',
                '*.key',
                '*.pem',
                '*.cer',
                '*.crt',
                
                # Lock files
                'package-lock.json',
                'yarn.lock',
                'poetry.lock',
                'Pipfile.lock',
                
                # IDE ve editor
                '.suo',
                '.user',
                '.userosscache',
                '.sln.docstates',
                '.vs',
                '.vscode'
            }

            # Path kontrolü
            if not repo_path.exists():
                raise Exception(f"Repository path does not exist: {repo_path}")
            
            # Koleksiyon adını normalize et
            normalized_name = repo_name.lower()
            collection_name = f"{normalized_name}_collection"
            print(f"Creating collection: {collection_name}")

            # Varsa eski koleksiyonu sil
            try:
                self.client.delete_collection(collection_name)
            except:
                pass

            # Yeni koleksiyon oluştur
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            self.collections[normalized_name] = collection

            documents = []
            metadatas = []
            ids = []
            doc_id = 0

            print("Scanning for files...")
            extensions = [
                # Web
                ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss", ".less",
                # Backend
                ".cs", ".cshtml", ".vb", ".php", ".py", ".java", ".rb",
                # Config/Data
                ".json", ".xml", ".yml", ".yaml", ".ini", ".config",
                # Documentation
                ".md", ".txt", ".rst",
                # Other
                ".sql", ".sh", ".bat", ".ps1"
            ]
            
            total_files = 0
            processed_files = 0
            skipped_files = 0
            
            for ext in extensions:
                for file in repo_path.rglob(f"*{ext}"):
                    # Klasör kontrolü
                    if any(ignored in file.parts for ignored in IGNORED_DIRS):
                        continue
                        
                    # Uzantı kontrolü
                    if any(file.name.endswith(ignored) for ignored in IGNORED_EXTENSIONS):
                        continue

                    total_files += 1
                    
                    if not any(p.startswith('.') for p in file.parts):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    chunks = self._split_content(content)
                                    update_progress(f"Processing {file.relative_to(repo_path)}")
                                    processed_files += 1
                                    for i, chunk in enumerate(chunks):
                                        # Benzersiz ID oluştur - tam dosya yolunu kullan
                                        file_path = str(file.relative_to(repo_path)).replace('\\', '/').replace('/', '_')
                                        ids.append(f"{repo_name}_{file_path}_{i}")
                                        documents.append(chunk)
                                        metadatas.append({
                                            "file": str(file.relative_to(repo_path)),
                                            "part": i + 1,
                                            "total_parts": len(chunks)
                                        })
                                        doc_id += 1
                        except Exception as e:
                            print(f"Error processing {file}: {str(e)}")
                            print(f"File encoding: {self.get_file_encoding(file)}")
                            skipped_files += 1
                            continue

            if documents:
                # Belgeleri daha küçük gruplar halinde ekle
                batch_size = 100
                total_batches = (len(documents) + batch_size - 1) // batch_size
                for i in range(0, len(documents), batch_size):
                    end = min(i + batch_size, len(documents))
                    update_progress(f"Adding batch {i//batch_size + 1} of {total_batches}")
                collection.add(
                        documents=documents[i:end],
                        metadatas=metadatas[i:end],
                        ids=ids[i:end]
                    )
            
            print(f"Added repository: {repo_name} with {len(documents)} chunks")
            
            # Update config
            for repo in self.config["repositories"]:
                if repo["name"] == repo_name:
                    repo["is_processed"] = True
                    break
            self.save_config()
            
            print(f"\nProcessing Summary:")
            print(f"Total files found: {total_files}")
            print(f"Successfully processed: {processed_files}")
            print(f"Skipped files: {skipped_files}")
            print(f"Total chunks created: {len(documents)}")

        except Exception as e:
            print(f"Error processing repository {repo_name}: {e}")
            # Hata durumunda koleksiyonu temizle
            try:
                if repo_name in self.collections:
                    self.client.delete_collection(collection_name)
                    del self.collections[repo_name]
            except:
                pass
            raise

    def _split_content(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Split content into chunks"""
        # Dosya boyutu limitini artır
        if len(content) > 5_000_000:  # 5MB
            print(f"Skipping large file: {len(content)} bytes")
            return []
        
        # Minimum chunk boyutu
        min_chunk_size = 500

        # Kod bloğu başlangıçlarını kontrol et
        def is_code_block_start(line: str) -> bool:
            patterns = [
                r'^\s*(public|private|protected)?\s*(class|interface|enum)\s+\w+',
                r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(',
                r'^\s*(function|def)\s+\w+\s*\(',
                r'^\s*namespace\s+\w+',
                # İç içe fonksiyonları da yakala
                r'^\s*\w+\s*=\s*function\s*\(',
                # Typescript/Javascript method tanımları
                r'^\s*\w+\s*:\s*function\s*\(',
                r'^\s*async\s+\w+\s*\(',
                # Python decoratorları
                r'^\s*@\w+',
            ]
            return any(re.match(pattern, line) for pattern in patterns)

        # Kod bloğu bitişini kontrol et
        def is_code_block_end(line: str, prev_lines: List[str]) -> bool:
            # Boş satır ve süslü parantez kontrolü
            if line.strip() == '}':
                return True
            # Python fonksiyon bitişi (boş satır + indent azalması)
            if line.strip() == '' and prev_lines:
                prev_indent = len(prev_lines[-1]) - len(prev_lines[-1].lstrip())
                curr_indent = len(line) - len(line.lstrip())
                return curr_indent < prev_indent
            return False

        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        in_code_block = False
        context_lines = 3  # Bağlam için önceki/sonraki satır sayısı
        
        for i, line in enumerate(lines):
            # Kod bloğu başlangıcını kontrol et
            if is_code_block_start(line):
                in_code_block = True
                # Önceki satırları da ekle (bağlam için)
                if i > 0:
                    current_chunk.extend(lines[max(0, i-context_lines):i])
                    current_size += sum(len(l) for l in lines[max(0, i-context_lines):i])

            # Chunk boyutunu kontrol et
            if current_size + len(line) > max_chunk_size and not in_code_block and current_size > min_chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += len(line)
            
            # Kod bloğu bitişini kontrol et
            if in_code_block and is_code_block_end(line, current_chunk):
                in_code_block = False
                # Sonraki satırları da ekle (bağlam için)
                if i < len(lines) - 1:
                    current_chunk.extend(lines[i+1:min(len(lines), i+1+context_lines)])
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    async def _query_ollama(self, prompt: str):
        """Stream response from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield f"data: {data['response']}\n\n"
                                elif "error" in data:
                                    yield f"data: Error: {data['error']}\n\n"
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"Error in _query_ollama: {e}")
            yield f"data: Error: {str(e)}\n\n"

    async def query(self, question: str, context: Optional[dict] = None) -> dict:
        try:
            print(f"Available collections: {list(self.collections.keys())}")
            if not self.collections:
                return "Henüz hiçbir repo eklenmemiş."

            # Context bilgilerini al
            query_type = context.get("type", "all")
            repo_name = context.get("repo", "").lower()  # Repo adını küçük harfe çevir
            files = context.get("files", [])

            print(f"Processing query - Type: {query_type}, Repo: {repo_name}")
            print(f"Question: {question}")

            collection = self.collections.get(repo_name)
            
            if not collection:
                print(f"Collection not found for repo: {repo_name}")
                return f"Repository '{repo_name}' not found in collections."

            try:
                if query_type == "files" and files:
                    # Dosya içeriklerini al
                    file_contents = []
                    for file_path in files:
                        results = collection.query(
                            query_texts=[file_path],
                            n_results=1,
                            where={"file": file_path}
                        )
                        if results["documents"][0]:
                            file_contents.extend(results["documents"][0])

                    if not file_contents:
                        return "No content found for selected files."

                    context_text = "\n---\n".join(file_contents)

                else:  # repo veya all için
                    # Repo içeriğinden ilgili kısımları al
                    results = collection.query(
                        query_texts=[question],
                        n_results=5,
                        include=["documents", "metadatas"]
                    )

                    if not results['documents'][0]:
                        return "Bu repo için ilgili bir bilgi bulunamadı."

                    context_parts = []
                    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                        context_parts.append(f"[{metadata['file']}]\n{doc}")

                    context_text = "\n---\n".join(context_parts)

                print(f"Found relevant content, generating response...")

                # Prompt'u hazırla
                if query_type == "files":
                    prompt = f"""You are analyzing these specific files from repository "{repo_name}":

{context_text}

Question: {question}

Please provide a clear and concise answer based on the file contents above."""
                elif query_type == "repo":
                    prompt = f"""You are analyzing the repository "{repo_name}". Here are some relevant parts:

{context_text}

Question: {question}

Please provide a clear and concise answer about this repository."""
                else:
                    prompt = f"""You are analyzing all repositories. Here are some relevant parts:

{context_text}

Question: {question}

Please provide a clear and concise answer."""

                return StreamingResponse(
                    self._query_ollama(prompt),
                    media_type='text/event-stream'
                )

            except Exception as e:
                print(f"Error in query: {e}")
                return f"Error processing query: {str(e)}"

        except Exception as e:
            print(f"Error: {e}")
            return str(e)

    @lru_cache(maxsize=32)
    def _get_cached_structure(self, repo_id: str, timestamp: int):
        """Get cached repository structure"""
        try:
            # Repo bilgilerini config'den al
            repo = next((r for r in self.config["repositories"] if r["name"] == repo_id), None)
            if not repo:
                raise ValueError(f"Repository {repo_id} not found")

            # Local path'i kullan
            repo_path = Path(repo["local_path"])
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repo_path}")

            def create_tree(path: Path, base_path: Path):
                """Recursively create file tree"""
                try:
                    if path.is_file():
                        if path.name.startswith('.'):
                            return None
                        return {
                            "type": "file",
                            "name": path.name,
                            "path": str(path.relative_to(base_path)).replace("\\", "/")
                        }

                    if path.is_dir():
                        if path.name.startswith('.') or path.name == '.git':
                            return None

                        children = []
                        with os.scandir(path) as entries:
                            items = sorted(entries, key=lambda x: (not x.is_file(), x.name.lower()))
                            for item in items:
                                child_path = Path(item.path)
                                child_tree = create_tree(child_path, base_path)
                                if child_tree:
                                    children.append(child_tree)

                        return {
                            "type": "folder",
                            "name": path.name,
                            "path": str(path.relative_to(base_path)).replace("\\", "/"),
                            "children": children
                        }

                    return None

                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    return None

            root = create_tree(repo_path, repo_path)
            if root:
                root["name"] = repo_id
            return root

        except Exception as e:
            print(f"Error getting project structure: {e}")
            raise HTTPException(status_code=404, detail="Repository not found")

    def get_repo_structure(self, repo_name: str) -> dict:
        """Get repository file structure"""
        try:
            # Cache kontrolü
            cache_key = f"structure_{repo_name}"
            current_time = time.time()
            
            # Cache'de varsa ve süresi geçmediyse kullan
            if cache_key in self._structure_cache:
                cached_data = self._structure_cache[cache_key]
                if current_time - cached_data['timestamp'] < self._cache_timeout:
                    return cached_data['structure']

            repo = next((r for r in self.config["repositories"] if r["name"] == repo_name), None)
            if not repo:
                raise Exception(f"Repository {repo_name} not found")

            repo_path = Path(repo["local_path"])
            if not repo_path.exists():
                raise Exception(f"Repository path {repo_path} does not exist")

            # İstenmeyen klasörleri ve dosyaları atla
            ignored_patterns = ['.git', '__pycache__', 'node_modules', '.idea', '.vscode']
            root = {'name': repo_name, 'type': 'folder', 'children': []}

            def create_folder_structure(current_path: Path, parent: dict):
                try:
                    # Klasörleri ve dosyaları sırala
                    items = sorted(current_path.iterdir(), 
                                 key=lambda x: (not x.is_dir(), x.name.lower()))
                    
                    for item in items:
                        # İstenmeyen klasörleri atla
                        if item.name in ignored_patterns:
                            continue
                            
                        # Gizli dosyaları atla
                        if item.name.startswith('.'):
                            continue
                            
                        if item.is_dir():
                            folder = {
                                'name': item.name,
                                'type': 'folder',
                                'children': []
                            }
                            parent['children'].append(folder)
                            create_folder_structure(item, folder)
                        else:
                            parent['children'].append({
                                'name': item.name,
                                'type': 'file',
                                'path': str(item.relative_to(repo_path)).replace('\\', '/')
                            })
                except Exception as e:
                    print(f"Error processing {current_path}: {e}")

            # Klasör yapısını oluştur
            create_folder_structure(repo_path, root)

            # Sonucu cache'le
            self._structure_cache[cache_key] = {
                'structure': root,
                'timestamp': current_time
            }

            return root

        except Exception as e:
            print(f"Error getting repo structure: {e}")
            raise

    def get_metrics(self, project_id: str):
        """Return project metrics"""
        try:
            repo = git.Repo(f"repos/{project_id}")
            return {
                "commits": len(list(repo.iter_commits())),
                "branches": len(repo.branches),
                "contributors": len(repo.git.shortlog("-s", "-n").split("\n")),
                "last_commit": repo.head.commit.committed_datetime.isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    def add_repo(self, repo_config, path=None):
        """
        Yeni bir repo ekle
        :param repo_config: Repo konfigürasyonu (dict)
        :param path: Repo local path (opsiyonel)
        """
        try:
            # Eğer path verilmemişse config'den al
            if path is None:
                path = repo_config.get("local_path")

            # Path'in geçerli olduğunu kontrol et
            if not Path(path).exists():
                raise ValueError(f"Path does not exist: {path}")

            # Process durumunu ekle
            repo_config["is_processed"] = False

            # Repoyu ekle
            if "repositories" not in self.config:
                self.config["repositories"] = []
            
            # Aynı isimli repo varsa güncelle
            for i, repo in enumerate(self.config["repositories"]):
                if repo["name"] == repo_config["name"]:
                    repo_config["is_processed"] = repo.get("is_processed", False)  # Mevcut durumu koru
                    self.config["repositories"][i] = repo_config
                    break
            else:
                # Yoksa yeni ekle
                self.config["repositories"].append(repo_config)
            
            # Konfigürasyonu kaydet
            self.save_config()
            
        except Exception as e:
            print(f"Error adding repo: {e}")
            raise

    def remove_repo(self, repo_name: str):
        """Repo'yu konfigürasyondan kaldır"""
        try:
            self.config["repositories"] = [
                r for r in self.config["repositories"] 
                if r["name"] != repo_name
            ]
            self.save_config()
        except Exception as e:
            print(f"Error removing repo: {e}")
            raise

    def _prepare_context(self, results, files=None):
        """Prepare context from search results"""
        context_parts = []
        
        if results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                context_parts.append(f"[{meta['file']}]\n{doc}")
            
        return "\n---\n".join(context_parts)

    def list_projects(self):
        """List all available repositories"""
        try:
            projects = []
            for repo in self.config["repositories"]:
                try:
                    repo_path = Path(repo["local_path"])
                    if repo_path.exists():
                        projects.append({
                            "name": repo["name"],
                            "path": str(repo_path),
                            "is_processed": repo.get("is_processed", False)
                        })
                except Exception as e:
                    print(f"Error checking repo {repo['name']}: {e}")
                    continue
            return projects
        except Exception as e:
            print(f"Error listing projects: {e}")
            return []

    def get_file_content(self, repo_name: str, file_path: str):
        """Get file content"""
        try:
            repo = next((r for r in self.config["repositories"] if r["name"] == repo_name), None)
            if not repo:
                raise HTTPException(status_code=404, detail="Repository not found")

            full_path = Path(repo["local_path"]) / file_path
            if not full_path.exists():
                raise HTTPException(status_code=404, detail="File not found")

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {
                    "content": content,
                    "language": self._detect_language(file_path),
                    "path": file_path
                }

            except UnicodeDecodeError:
                return {
                    "error": "Binary file cannot be displayed"
                }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.js': 'javascript',
            '.jsx': 'jsx',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.py': 'python',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.json': 'json',
            '.md': 'markdown',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.sh': 'bash',
            '.bash': 'bash',
            '.sql': 'sql',
            '.php': 'php',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.dart': 'dart',
        }
        return language_map.get(ext, 'plaintext')

    async def query_all(self, question: str):
        """Query across all repositories"""
        try:
            prompt = f"""Please provide a concise answer based on the available code:
Q: {question}"""
            
            async for chunk in self._query_ollama(prompt):
                yield chunk
            
        except Exception as e:
            yield f"Error: {str(e)}"

    async def query_repo(self, repo_id: str, question: str):
        """Query specific repository"""
        try:
            print(f"Querying repo {repo_id} with question: {question}")
            collection_name = self._normalize_collection_name(repo_id)
            print(f"Collection name: {collection_name}")
            
            # Önce repo'nun işlenip işlenmediğini kontrol et
            repo = next((r for r in self.config["repositories"] if r["name"] == repo_id), None)
            if not repo:
                raise Exception(f"Repository {repo_id} not found in config")
                
            if not repo.get("is_processed"):
                raise Exception(f"Repository {repo_id} has not been processed yet. Please process it first from the admin panel.")
            
            collection = self.collections.get(collection_name)
            print(f"Got collection: {collection}")
            
            results = collection.query(
                query_texts=[question],
                n_results=3,
                include=["documents", "metadatas"]
            )
            print(f"Query results: {results}")
            
            if not results['documents'][0]:
                raise Exception(f"No content found in repository {repo_id}. Try processing it again from the admin panel.")
            
            context = self._prepare_context(results)
            print(f"Prepared context: {context}")
            
            prompt = f"""Please provide a concise answer based on the code:
{context}
Q: {question}"""
            
            async for chunk in self._query_ollama(prompt):
                yield chunk
            
        except Exception as e:
            print(f"Error in query_repo: {str(e)}")
            yield f"Error: {str(e)}"

    async def query_files(self, repo_id: str, files: List[str], question: str):
        """Query specific files in a repository"""
        try:
            repo = next((r for r in self.config["repositories"] if r["name"] == repo_id), None)
            if not repo:
                raise Exception(f"Repository {repo_id} not found in config")
            
            repo_path = Path(repo["local_path"])
            print(f"Querying files in repo {repo_id}: {files}")
            
            content = []
            max_content_size = 8000  # Maksimum context boyutu
            current_size = 0
            
            for file_path in files:
                try:
                    full_path = repo_path / file_path
                    print(f"Checking file: {full_path}")
                    
                    if not full_path.exists():
                        print(f"File not found: {full_path}")
                        continue
                        
                    if not full_path.is_file():
                        print(f"Not a file: {full_path}")
                        continue
                        
                    # JSON dosyaları için özel işleme
                    if file_path.endswith('.json'):
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                json_content = json.load(f)
                                file_content = json.dumps(json_content, indent=2)
                        except json.JSONDecodeError as je:
                            print(f"Invalid JSON in {file_path}: {je}")
                            continue
                    else:
                        # Normal text dosyaları için
                        with open(full_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()

                    # Dosya çok büyükse chunk'la
                    if len(file_content) > 2000:
                        chunks = self._split_content(file_content)
                        file_content = "\n\n".join(chunks[:3])  # İlk 3 chunk'ı al
                    
                    # Context boyutunu kontrol et
                    if current_size + len(file_content) > max_content_size:
                        continue
                    
                    content.append(f"File: {file_path}\n\n{file_content}")
                    current_size += len(file_content)
                    print(f"Successfully read file: {file_path}")
                    
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    continue

            if not content:
                raise Exception("No readable files found")

            # Dosya içeriklerini birleştir
            context = "\n\n---\n\n".join(content)

            # Prompt oluştur
            prompt = f"""Please analyze these files and answer the question.
For JSON files, explain the key configurations and dependencies.
For code files, explain the main functionality.

{context}

Question: {question}"""
            
            async for chunk in self._query_ollama(prompt):
                yield chunk
            
        except Exception as e:
            yield f"Error: {str(e)}"

    def get_file_history(self, project_id: str, file_path: str):
        """Dosya commit geçmişini getir"""
        try:
            repo_path = Path(f"repos/{project_id}")
            if not repo_path.exists():
                return {"error": "Repo bulunamadı"}

            repo = git.Repo(repo_path)
            file_path = Path(file_path)
            full_path = repo_path / file_path

            if not full_path.exists():
                return {"error": "Dosya bulunamadı"}

            commits = []
            try:
                # Git log komutunu kullan
                log_info = repo.git.log(
                    '--follow',  # Dosya yeniden adlandırılmış olsa bile takip et
                    '--pretty=format:%H|%an|%at|%s',  # Hash|Author|Timestamp|Message
                    '--',  # Dosya yolunu belirt
                    str(file_path)
                )

                for line in log_info.split('\n'):
                    if not line.strip():
                        continue
                        
                    hash_val, author, timestamp, message = line.split('|')
                    
                    # Değişiklikleri al
                    try:
                        stats = repo.git.show(
                            '--numstat',
                            '--format=""',
                            hash_val,
                            '--',
                            str(file_path)
                        ).strip()
                        
                        if stats:
                            additions, deletions, _ = stats.split('\n')[0].split('\t')
                        else:
                            additions, deletions = 0, 0
                    except:
                        additions, deletions = 0, 0

                    commits.append({
                        "hash": hash_val,
                        "author": author,
                        "date": datetime.fromtimestamp(int(timestamp)).isoformat(),
                        "message": message.strip(),
                        "changes": {
                            "additions": additions,
                            "deletions": deletions
                        }
                    })

            except git.exc.GitCommandError as e:
                print(f"Git log error: {e}")
                return {"error": "Dosya geçmişi alınamadı"}

            return {
                "file": str(file_path),
                "commits": commits,
                "total_commits": len(commits)
            }
            
        except Exception as e:
            print(f"Error in get_file_history: {e}")
            return {"error": str(e)}

    def get_chat_history(self, project_id: str):
        """Proje için sohbet geçmişini getir"""
        try:
            return self.chat_history.get(project_id, [])
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return []

    def reset_collections(self):
        """Tüm koleksiyonları temizle ve yeniden oluştur"""
        try:
            # Mevcut koleksiyonları listele ve sil
            collections = self.client.list_collections()
            for collection in collections:
                try:
                    self.client.delete_collection(collection.name)
                    print(f"Deleted collection: {collection.name}")
                except Exception as e:
                    print(f"Error deleting collection {collection.name}: {e}")
            
            # Koleksiyonlar sözlüğünü temizle
            self.collections = {}
            print("All collections reset")
            
        except Exception as e:
            print(f"Error resetting collections: {e}")

    def list_repos(self):
        """Return list of repositories"""
        try:
            # repos.yaml dosyasından repo listesini al
            with open("repos.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("repositories", [])
        except Exception as e:
            print(f"Error listing repositories: {e}")
            return []

    def _load_existing_collections(self):
        """Mevcut koleksiyonları yükle"""
        try:
            # Önce config'deki repoları kontrol et
            print("\nChecking repositories in config:")
            for repo in self.config["repositories"]:
                print(f"- {repo['name']}: {'processed' if repo.get('is_processed') else 'not processed'}")

            # ChromaDB'den koleksiyonları al
            collection_names = self.client.list_collections()
            print(f"\nFound collections in ChromaDB: {collection_names}")

            for name in collection_names:
                try:
                    # Koleksiyon adından repo adını çıkar
                    # Koleksiyon adını normalize et
                    repo_name = name.replace('_collection', '')
                    
                    loaded_collection = self.client.get_collection(
                        name=name,
                        embedding_function=self.embedding_fn
                    )
                    # Koleksiyonları küçük harfle sakla
                    self.collections[repo_name.lower()] = loaded_collection
                    print(f"Loaded collection: {name} for repository: {repo_name}")
                except Exception as e:
                    print(f"Error loading collection {name}: {e}")

        except Exception as e:
            print(f"Error listing collections: {e}")

        print(f"\nLoaded collections: {list(self.collections.keys())}")

    def set_context(self, context_type: str, repo: str = None, files: List[str] = None):
        """UI context'ini güncelle"""
        try:
            if context_type not in ["all", "repo", "files"]:
                raise ValueError("Invalid context type")
                
            self.current_context = {
                "type": context_type,
                "repo": repo,
                "files": files or []
            }
            
            return {
                "status": "success",
                "context": self.current_context,
                "message": self._get_context_message()
            }
            
        except Exception as e:
            print(f"Error setting context: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _get_context_message(self) -> str:
        """Mevcut context için kullanıcı mesajı oluştur"""
        if self.current_context["type"] == "all":
            return "🔍 Searching across all repositories"
        elif self.current_context["type"] == "repo":
            return f"📁 Searching in {self.current_context['repo']}"
        else:
            files = self.current_context["files"]
            file_count = len(files)
            return f"📄 Searching in {file_count} selected file{'s' if file_count > 1 else ''}"

    def get_context_info(self):
        """Mevcut context bilgisini döndür"""
        return {
            "current": self.current_context,
            "message": self._get_context_message(),
            "available_repos": self.list_repos(),
            "available_files": self._get_available_files() if self.current_context["repo"] else []
        }

    def _get_available_files(self):
        """Get available files for current repo"""
        if not self.current_context["repo"]:
            return []
            
        try:
            structure = self.get_repo_structure(self.current_context["repo"])
            files = []
            
            def extract_files(node):
                if node["type"] == "file":
                    files.append({
                        "path": node["path"],
                        "name": node["name"],
                        "selected": node["path"] in (self.current_context["files"] or [])
                    })
                elif node["type"] == "folder" and "children" in node:
                    for child in node["children"]:
                        extract_files(child)
                        
            extract_files(structure)
            return files
            
        except Exception as e:
            print(f"Error getting available files: {e}")
            return []

    async def query_with_context(self, question: str):
        """Mevcut context'e göre sorguyu yönlendir"""
        try:
            context = self.current_context
            
            if context["type"] == "all":
                return await self.query_all(question)
            elif context["type"] == "repo" and context["repo"]:
                return await self.query_repo(context["repo"], question)
            elif context["type"] == "files" and context["files"]:
                return await self.query_files(context["repo"], context["files"], question)
            else:
                return "Please select a search context and make sure you have selected files or a repository."
                
        except Exception as e:
            return f"Error: {str(e)}"

    def get_file_encoding(self, file_path):
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw = f.read()
                result = chardet.detect(raw)
                return result['encoding']
        except:
            return "unknown"

    async def stream_response(self, prompt: str):
        """Stream the response from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield f"data: {data['response']}\n\n"
                                elif "error" in data:
                                    yield f"data: Error: {data['error']}\n\n"
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"Error in stream_response: {e}")
            yield f"data: Error: {str(e)}\n\n"

class LLMAssistant:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # GPU kullanımını kontrol et ve detaylı log al
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch device: {self.device}")
        
        if self.device == "cuda":
            # GPU belleğini temizle
            torch.cuda.empty_cache()
            # GPU'yu ısıt
            self._warmup_gpu()
        
        # Model GPU'ya taşı
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model = model.to(self.device)
        
        # Embedding fonksiyonunu GPU ile kullan
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=self.device
        )
        
        self.collections = {}
        self._load_existing_collections()

    def _warmup_gpu(self):
        """GPU'yu ısıt"""
        if self.device == "cuda":
            try:
                # Küçük bir model yükleyerek GPU'yu hazırla
                model = torch.nn.Linear(100, 100).cuda()
                x = torch.randn(1, 100).cuda()
                for _ in range(10):
                    model(x)
                del model, x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"GPU warmup error: {e}")

    def _load_existing_collections(self):
        try:
            collection_names = self.client.list_collections()
            for name in collection_names:
                try:
                    # Koleksiyon adından repo adını çıkar
                    repo_name = name.replace('_collection', '').title()
                    
                    loaded_collection = self.client.get_collection(
                        name=name,
                        embedding_function=self.embedding_fn
                    )
                    self.collections[repo_name] = loaded_collection
                    print(f"Loaded collection: {name} for repository: {repo_name}")
                except Exception as e:
                    print(f"Error loading collection {name}: {e}")

        except Exception as e:
            print(f"Error listing collections: {e}")

    def add_repo(self, name: str, path: Path):
        try:
            # Önce mevcut koleksiyonu temizle
            try:
                # Koleksiyonu bulmaya çalış
                existing = self.client.get_collection(name=name)
                if existing:
                    self.client.delete_collection(name=name)
                    if name in self.collections:
                        del self.collections[name]
                    print(f"Deleted existing collection: {name}")
            except Exception as e:
                # Koleksiyon yoksa sorun değil
                print(f"No existing collection found for {name}")
            
            # Yeni koleksiyon oluştur
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_fn
            )
            
            documents = []
            metadatas = []
            ids = []
            doc_id = 0
            
            # Dosyaları küçük parçalara böl
            for ext in [".py", ".js", ".jsx", ".ts", ".tsx", ".md"]:
                for file in path.rglob(f"*{ext}"):
                    if not any(p.startswith('.') for p in file.parts):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    # Dosyayı parçalara böl
                                    chunks = self._split_content(content)
                                    for i, chunk in enumerate(chunks):
                                        documents.append(chunk)
                                        metadatas.append({
                                            "file": str(file.relative_to(path)),
                                            "part": i + 1,
                                            "total_parts": len(chunks)
                                        })
                                        ids.append(f"{file.stem}_chunk_{i}")
                                        doc_id += 1
                        except Exception as e:
                            print(f"Error processing {file}: {str(e)}")
                            print(f"File encoding: {self.get_file_encoding(file)}")
                            continue

            if documents:
                # Belgeleri daha küçük gruplar halinde ekle
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    end = min(i + batch_size, len(documents))
                    collection.add(
                        documents=documents[i:end],
                        metadatas=metadatas[i:end],
                        ids=ids[i:end]
                    )
            
            self.collections[name] = collection
            print(f"Added repository: {name} with {len(documents)} chunks")
            
        except Exception as e:
            print(f"Error adding repository {name}: {e}")
            # Hata durumunda koleksiyonu temizle
            try:
                if name in self.collections:
                    self.client.delete_collection(name)
                    del self.collections[name]
            except:
                pass
            raise

    def _split_content(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Split content into chunks"""
        # Dosya boyutu limitini artır
        if len(content) > 5_000_000:  # 5MB
            print(f"Skipping large file: {len(content)} bytes")
            return []
        
        # Minimum chunk boyutu
        min_chunk_size = 500

        # Kod bloğu başlangıçlarını kontrol et
        def is_code_block_start(line: str) -> bool:
            patterns = [
                r'^\s*(public|private|protected)?\s*(class|interface|enum)\s+\w+',
                r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(',
                r'^\s*(function|def)\s+\w+\s*\(',
                r'^\s*namespace\s+\w+',
                # İç içe fonksiyonları da yakala
                r'^\s*\w+\s*=\s*function\s*\(',
                # Typescript/Javascript method tanımları
                r'^\s*\w+\s*:\s*function\s*\(',
                r'^\s*async\s+\w+\s*\(',
                # Python decoratorları
                r'^\s*@\w+',
            ]
            return any(re.match(pattern, line) for pattern in patterns)

        # Kod bloğu bitişini kontrol et
        def is_code_block_end(line: str, prev_lines: List[str]) -> bool:
            # Boş satır ve süslü parantez kontrolü
            if line.strip() == '}':
                return True
            # Python fonksiyon bitişi (boş satır + indent azalması)
            if line.strip() == '' and prev_lines:
                prev_indent = len(prev_lines[-1]) - len(prev_lines[-1].lstrip())
                curr_indent = len(line) - len(line.lstrip())
                return curr_indent < prev_indent
            return False

        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        in_code_block = False
        context_lines = 3  # Bağlam için önceki/sonraki satır sayısı
        
        for i, line in enumerate(lines):
            # Kod bloğu başlangıcını kontrol et
            if is_code_block_start(line):
                in_code_block = True
                # Önceki satırları da ekle (bağlam için)
                if i > 0:
                    current_chunk.extend(lines[max(0, i-context_lines):i])
                    current_size += sum(len(l) for l in lines[max(0, i-context_lines):i])

            # Chunk boyutunu kontrol et
            if current_size + len(line) > max_chunk_size and not in_code_block and current_size > min_chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += len(line)
            
            # Kod bloğu bitişini kontrol et
            if in_code_block and is_code_block_end(line, current_chunk):
                in_code_block = False
                # Sonraki satırları da ekle (bağlam için)
                if i < len(lines) - 1:
                    current_chunk.extend(lines[i+1:min(len(lines), i+1+context_lines)])
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    async def query(self, question: str, context: Optional[dict] = None) -> dict:
        try:
            print(f"Available collections: {list(self.collections.keys())}")
            if not self.collections:
                return "Henüz hiçbir repo eklenmemiş."

            # Context bilgilerini al
            query_type = context.get("type", "all")
            repo_name = context.get("repo", "").lower()  # Repo adını küçük harfe çevir
            files = context.get("files", [])

            print(f"Processing query - Type: {query_type}, Repo: {repo_name}")
            print(f"Question: {question}")

            collection = self.collections.get(repo_name)
            
            if not collection:
                print(f"Collection not found for repo: {repo_name}")
                return f"Repository '{repo_name}' not found in collections."

            try:
                if query_type == "files" and files:
                    # Dosya içeriklerini al
                    file_contents = []
                    for file_path in files:
                        results = collection.query(
                            query_texts=[file_path],
                            n_results=1,
                            where={"file": file_path}
                        )
                        if results["documents"][0]:
                            file_contents.extend(results["documents"][0])

                    if not file_contents:
                        return "No content found for selected files."

                    context_text = "\n---\n".join(file_contents)

                else:  # repo veya all için
                    # Repo içeriğinden ilgili kısımları al
                    results = collection.query(
                        query_texts=[question],
                        n_results=5,
                        include=["documents", "metadatas"]
                    )

                    if not results['documents'][0]:
                        return "Bu repo için ilgili bir bilgi bulunamadı."

                    context_parts = []
                    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                        context_parts.append(f"[{metadata['file']}]\n{doc}")

                    context_text = "\n---\n".join(context_parts)

                print(f"Found relevant content, generating response...")

                # Prompt'u hazırla
                if query_type == "files":
                    prompt = f"""You are analyzing these specific files from repository "{repo_name}":

{context_text}

Question: {question}

Please provide a clear and concise answer based on the file contents above."""
                elif query_type == "repo":
                    prompt = f"""You are analyzing the repository "{repo_name}". Here are some relevant parts:

{context_text}

Question: {question}

Please provide a clear and concise answer about this repository."""
                else:
                    prompt = f"""You are analyzing all repositories. Here are some relevant parts:

{context_text}

Question: {question}

Please provide a clear and concise answer."""

                return StreamingResponse(
                    self._query_ollama(prompt),
                    media_type='text/event-stream'
                )

            except Exception as e:
                print(f"Error in query: {e}")
                return f"Error processing query: {str(e)}"

        except Exception as e:
            print(f"Error: {e}")
            return str(e)

def initialize_system():
    try:
        print("Initializing system...")
        
        # Repo konfigürasyonunu yükle
        with open("repos.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        # Servisleri oluştur
        repo_manager = RepoManager()
        project_assistant = ProjectAssistant()

        print("\nSystem initialized - Use admin panel to manage repos")
        return project_assistant, repo_manager  # Her iki servisi de döndür
        
    except Exception as e:
        print(f"Error during system initialization: {e}")
        raise 