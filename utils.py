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
from starlette.responses import FileResponse
import asyncio
import requests

class RepoManager:
    def __init__(self, base_path: str = "repos"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def clone_or_pull_repo(self, repo_name: str) -> Path:
        """Repoyu klonla veya güncelle"""
        try:
            # Repo bilgilerini al
            with open("repos.yaml", "r") as f:
                config = yaml.safe_load(f)
                repo_info = next(r for r in config["repositories"] if r["name"] == repo_name)

            repo_path = self.base_path / repo_name
            
            if repo_path.exists():
                print(f"Pulling {repo_name}...")
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
            else:
                print(f"Cloning {repo_name}...")
                git.Repo.clone_from(repo_info["url"], repo_path)
                
            return repo_path

        except Exception as e:
            print(f"Error in clone_or_pull_repo: {str(e)}")
            raise

    def delete_repo(self, repo_name: str):
        """Repoyu sil"""
        try:
            repo_path = self.base_path / repo_name
            
            # Önce git repo klasörünü sil
            if repo_path.exists():
                import shutil
                shutil.rmtree(repo_path)
            
            # Sonra repos.yaml'dan kaldır
            with open("repos.yaml", "r") as f:
                config = yaml.safe_load(f)
                
            config["repositories"] = [
                r for r in config["repositories"] 
                if r["name"] != repo_name
            ]
            
            with open("repos.yaml", "w") as f:
                yaml.dump(config, f)
                
        except Exception as e:
            print(f"Error in delete_repo: {str(e)}")
            raise

    def pull_repo(self, repo_name: str):
        """Repoyu güncelle"""
        try:
            repo_path = self.base_path / repo_name
            if repo_path.exists():
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
            else:
                raise Exception(f"Repo {repo_name} does not exist locally")
        except Exception as e:
            print(f"Error in pull_repo: {str(e)}")
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
        self.ollama_url = "http://localhost:11434"
        self.max_retries = 5
        self.timeout = 60.0
        self._check_ollama_connection()

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

    def process_repo(self, repo_name: str, repo_path: Path):
        """Process repository content and create collection"""
        try:
            # Delete existing collection if exists
            try:
                self.client.delete_collection(repo_name)
                print(f"Deleted existing collection: {repo_name}")
            except:
                pass

            # Create new collection
            collection = self.client.create_collection(
                name=repo_name,
                embedding_function=self.embedding_fn
            )
            self.collections[repo_name] = collection

            # Scan repository content
            documents = []
            metadatas = []
            ids = []
            doc_id = 0

            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        # Read text files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Split content into chunks
                        chunks = self._split_content(content)
                        
                        # Add each chunk to collection
                        for i, chunk in enumerate(chunks):
                            relative_path = str(file_path.relative_to(repo_path))
                            documents.append(chunk)
                            metadatas.append({
                                "file": relative_path,
                                "chunk": i,
                                "total_chunks": len(chunks)
                            })
                            # Create unique ID
                            unique_id = f"{repo_name}_{relative_path.replace('/', '_')}_{doc_id}"
                            ids.append(unique_id)
                            doc_id += 1
                            
                            # Add in batches for memory management
                            if len(documents) >= 100:
                                collection.add(
                                    documents=documents,
                                    metadatas=metadatas,
                                    ids=ids
                                )
                                documents = []
                                metadatas = []
                                ids = []
                                
                    except UnicodeDecodeError:
                        # Skip binary files
                        continue
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

            # Add remaining documents
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added {doc_id} documents to collection {repo_name}")

        except Exception as e:
            print(f"Error processing repo {repo_name}: {e}")
            raise

    def _split_content(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Split content into chunks"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += len(line)
            
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    async def _query_ollama(self, prompt: str):
        """Stream-enabled Ollama request"""
        last_response = ""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "num_ctx": 4096,
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "num_gpu": 1,
                            "gpu_layers": 45,
                            "mmap": True,
                            "num_thread": 8,
                            "repeat_penalty": 1.2,
                            "num_predict": 200
                        }
                    }
                ) as response:
                    if response.status_code != 200:
                        raise Exception(f"Ollama API error: {response.status_code}")
                        
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                if "response" in data:
                                    new_text = data["response"]
                                    # Remove prefixes like "Answer:", "Short Answer:"
                                    new_text = new_text.replace("Answer: ", "")
                                    new_text = new_text.replace("Short Answer: ", "")
                                    if new_text != last_response:
                                        last_response = new_text
                                        yield new_text
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            yield f"Error: {str(e)}"

    async def query(self, project_id: str, question: str):
        try:
            collection = self.client.get_collection(
                name=project_id,
                embedding_function=self.embedding_fn
            )

            # Get fewer results
            results = collection.query(
                query_texts=[question],
                n_results=2,
                include=["documents", "metadatas"]
            )

            context = self._prepare_context(results, [])
            
            # Shorter prompt
            prompt = f"""Please provide a concise answer:
{context}
Q: {question}"""

            answer = await self._query_ollama(prompt)
            
            return {
                "answer": answer,
                "context_type": "single_repo",
                "context": context
            }

        except Exception as e:
            print(f"Error in query: {e}")
            raise

    def get_project_structure(self, project_id: str):
        """Return project directory structure"""
        try:
            repo_path = Path(f"repos/{project_id}")
            if not repo_path.exists():
                return {"error": "Repository not found"}
            
            structure = {"name": project_id, "type": "folder", "children": []}
            
            def build_tree(path, node):
                try:
                    for item in path.iterdir():
                        # Skip hidden files and folders
                        if item.name.startswith('.'):
                            continue
                        
                        if item.is_file():
                            # Make file path platform independent
                            relative_path = str(item.relative_to(repo_path)).replace(os.sep, '/')
                            node["children"].append({
                                "name": item.name,
                                "type": "file",
                                "path": relative_path
                            })
                        else:
                            child = {
                                "name": item.name,
                                "type": "folder",
                                "children": []
                            }
                            build_tree(item, child)
                            node["children"].append(child)
                except Exception as e:
                    print(f"Error in build_tree for {path}: {e}")
            
            build_tree(repo_path, structure)
            return structure
            
        except Exception as e:
            print(f"Error in get_project_structure: {e}")
            return {"error": str(e)}

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

    def add_repo(self, name: str, path: Path):
        """Add new repository"""
        try:
            # First clear existing collection
            try:
                existing = self.client.get_collection(name=name)
                if existing:
                    self.client.delete_collection(name=name)
                    if name in self.collections:
                        del self.collections[name]
                    print(f"Deleted existing collection: {name}")
            except Exception as e:
                print(f"No existing collection found for {name}")
            
            # Create new collection
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_fn
            )
            
            # Scan and add files
            documents = []
            metadatas = []
            ids = []
            doc_id = 0
            
            for ext in [".py", ".js", ".jsx", ".ts", ".tsx", ".md"]:
                for file in path.rglob(f"*{ext}"):
                    if not any(p.startswith('.') for p in file.parts):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    chunks = self._split_content(content)
                                    for i, chunk in enumerate(chunks):
                                        documents.append(chunk)
                                        metadatas.append({
                                            "file": str(file.relative_to(path)),
                                            "part": i + 1,
                                            "total_parts": len(chunks)
                                        })
                                        ids.append(f"{name}_{doc_id}")
                                        doc_id += 1
                        except Exception as e:
                            print(f"Error reading {file}: {e}")

            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            self.collections[name] = collection
            print(f"Added repository: {name} with {len(documents)} chunks")
            
        except Exception as e:
            print(f"Error adding repository {name}: {e}")
            raise

    def _prepare_context(self, results, files=None):
        """Prepare context from search results"""
        context_parts = []
        
        if results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                context_parts.append(f"[{meta['file']}]\n{doc}")
            
        return "\n---\n".join(context_parts)

    def list_projects(self) -> List[str]:
        """Mevcut projeleri listele"""
        try:
            projects = list(self.collections.keys())
            print(f"Available collections: {projects}")  # Debug log
            return projects
        except Exception as e:
            print(f"Error listing projects: {e}")
            return []

    def get_file_content(self, project_id: str, file_path: str):
        """Dosya içeriğini getir"""
        try:
            # URL decode ve path normalizasyonu
            file_path = Path(file_path.replace('/', os.sep))
            full_path = Path(f"repos/{project_id}") / file_path
            
            print(f"Trying to access: {full_path}")  # Debug için
            
            if not full_path.exists():
                print(f"File not found: {full_path}")  # Debug için
                return {"error": "Dosya bulunamadı"}
            
            if not full_path.is_file():
                return {"error": "Bu bir dosya değil"}
            
            # Dizin traversal güvenliği
            try:
                full_path.relative_to(f"repos/{project_id}")
            except ValueError:
                return {"error": "Geçersiz dosya yolu"}
            
            # Resim dosyaları için
            if full_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg']:
                return FileResponse(full_path)
            
            # Metin dosyaları için
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return {"error": "Bu dosya türü desteklenmiyor"}
            
            return {
                "content": content,
                "language": self._detect_language(str(file_path))
            }
        except Exception as e:
            print(f"Error in get_file_content: {e}")  # Debug için
            return {"error": str(e)}
        
    def _detect_language(self, file_path: str) -> str:
        """Dosya uzantısına göre dil tespiti yap"""
        ext = Path(file_path).suffix.lower()
        languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json'
        }
        return languages.get(ext, 'plaintext')

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
            collection = self.client.get_collection(
                name=repo_id,
                embedding_function=self.embedding_fn
            )
            
            results = collection.query(
                query_texts=[question],
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            context = self._prepare_context(results)
            prompt = f"""Please provide a concise answer based on the code:
{context}
Q: {question}"""
            
            async for chunk in self._query_ollama(prompt):
                yield chunk
            
        except Exception as e:
            yield f"Error: {str(e)}"

    async def query_files(self, repo_id: str, files: List[str], question: str):
        """Query specific files in a repository"""
        try:
            collection = self.client.get_collection(
                name=repo_id,
                embedding_function=self.embedding_fn
            )
            
            results = collection.query(
                query_texts=[question],
                n_results=3,
                where={"file": {"$in": files}},
                include=["documents", "metadatas"]
            )
            
            context = self._prepare_context(results, files)
            prompt = f"""Please provide a concise answer based on the selected files:
{context}
Q: {question}"""
            
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
            # Yeni API'ye göre önce koleksiyon isimlerini al
            collection_names = self.client.list_collections()
            for name in collection_names:
                try:
                    # Her koleksiyonu ayrı ayrı yükle
                    collection = self.client.get_collection(name=name)
                    self.collections[name] = collection
                    print(f"Loaded existing collection: {name}")
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
                            print(f"Error reading {file}: {e}")

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
        # Chunk boyutunu GPU için artırdık
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += len(line)
            
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    async def query(self, question: str, context: Optional[str] = None) -> dict:
        try:
            if not self.collections:
                return {
                    "question": question,
                    "answer": "Henüz hiçbir repo eklenmemiş.",
                    "context": context,
                    "confidence": 0.0
                }

            repo_name = context if context in self.collections else list(self.collections.keys())[0]
            collection = self.collections[repo_name]

            try:
                # Daha az sonuç al
                results = collection.query(
                    query_texts=[question],
                    n_results=3,  # 5'ten 3'e düşürdük
                    include=["documents", "metadatas"]
                )

                if not results['documents'][0]:
                    return {
                        "question": question,
                        "answer": "Bu repo için ilgili bir bilgi bulunamadı.",
                        "context": repo_name,
                        "confidence": 0.0
                    }

                # Bağlamı daha kısa tut
                context_parts = []
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    context_parts.append(f"[{metadata['file']}]\n{doc[:500]}")  # Her parçayı 500 karakterle sınırla

                context_text = "\n---\n".join(context_parts)

                print(f"Found {len(context_parts)} relevant chunks")
                
                # GPU kullanımını kontrol et
                print("Sending request to Ollama with GPU config:")
                print(f"GPU Layers: 45")
                print(f"Num GPU: 1")
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "codellama",
                            "prompt": f"""Kısa yanıt ver: {question}""",
                            "stream": True,
                            "options": {
                                "num_ctx": 512,
                                "temperature": 0.1,
                                "gpu_layers": 45,     # Maksimum GPU layer
                                "num_gpu": 1,
                                "mmap": True,         # GPU bellek yönetimi
                                "f16": True,          # 16-bit floating point
                                "numa": True,         # NUMA optimizasyonu
                                "threads": 8          # Thread sayısı
                            }
                        }
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"Ollama API error: {response.text}")

                    result = response.json()
                    if "error" in result:
                        raise Exception(f"Ollama error: {result['error']}")

                    return {
                        "question": question,
                        "answer": result.get("response", "Bir hata oluştu"),
                        "context": repo_name,
                        "confidence": 0.8 if result.get("response") else 0.0
                    }

            except Exception as e:
                print(f"Query processing error: {str(e)}")
                raise

        except Exception as e:
            print(f"Error during query: {str(e)}")
            return {
                "question": question,
                "answer": f"Bir hata oluştu: {str(e)}",
                "context": context,
                "confidence": 0.0
            }

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