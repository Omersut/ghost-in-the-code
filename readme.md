# 👻 Ghost in the Code

An AI-powered code exploration and documentation assistant that helps developers understand codebases through natural language conversations.

## ✨ Features

- 🤖 Natural language interactions with your codebase
- 📁 Support for multiple repositories
- 🔍 Smart code search and context-aware responses
- 📝 File content preview with syntax highlighting
- 🌲 Interactive file explorer
- 🔄 Git integration (clone, pull, process repositories)
- 🎨 Modern dark theme UI
- ⚡ Real-time streaming responses
- 📊 File history and commit tracking
- 🔍 Multi-file search capabilities
- 🧠 Context-aware code understanding
- 🚀 GPU acceleration support

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- [Ollama](https://ollama.ai/) with Mistral model installed
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
git clone https://github.com/Omersut/ghost-in-the-code.git
cd ghost-in-the-code

2. Install Python dependencies:
pip install -r requirements.txt

3. Start Ollama service:
ollama serve

4. Pull the Mistral model:
ollama pull mistral

5. Run the application:
uvicorn main:app --reload

6. Open http://localhost:8000 in your browser

### Configuration

Create a `repos.yaml` file in the project root:
repositories:
  - name: my-project
    url: https://github.com/username/repo.git
    branch: main
    description: Project description

## 🛠️ Technology Stack

- **Backend**: 
  - FastAPI for API server
  - ChromaDB for vector storage
  - Sentence Transformers for embeddings
  - GitPython for repository management
- **LLM**: 
  - Ollama (Mistral) for natural language processing
  - CUDA acceleration support
- **Frontend**: 
  - Vanilla JavaScript
  - Modern CSS with dark theme
  - Real-time updates
- **Database**: 
  - ChromaDB for efficient vector storage
  - File-based repository configuration
- **Version Control**: 
  - Git integration for repository management
  - Commit history tracking

## 🔧 Architecture

The application follows a modular client-server architecture:

1. **Vector Store Layer**:
   - Uses ChromaDB for efficient storage and retrieval of code embeddings
   - Handles document chunking and embedding generation
   - Maintains separate collections for each repository

2. **LLM Integration Layer**:
   - Connects to Ollama for natural language understanding
   - Handles prompt engineering and context management
   - Supports streaming responses for real-time interaction

3. **Repository Management Layer**:
   - Manages Git operations (clone, pull, delete)
   - Tracks repository status and updates
   - Handles file system operations

4. **API Layer**:
   - RESTful endpoints for all operations
   - Real-time communication using Server-Sent Events
   - Error handling and status reporting

5. **Frontend Layer**:
   - Modern, responsive UI with dark theme
   - Interactive file explorer
   - Real-time chat interface
   - File preview with syntax highlighting

## 💡 Usage Examples

1. **Repository Management**:
# Add a new repository
curl -X POST http://localhost:8000/admin/repos \
  -H "Content-Type: application/json" \
  -d '{"name":"my-project","url":"https://github.com/user/repo.git"}'

2. **Code Query**:
# Query specific repository
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does the authentication work?","project_id":"my-project"}'

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for the LLM integration
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 🐛 Troubleshooting

1. **Ollama Connection Issues**:
   - Ensure Ollama service is running (`ollama serve`)
   - Check if Mistral model is installed (`ollama list`)

2. **GPU Issues**:
   - Verify CUDA installation (`nvidia-smi`)
   - Check GPU memory usage
   - Adjust batch sizes if needed

3. **Repository Issues**:
   - Verify Git installation
   - Check repository permissions
   - Ensure valid repository URLs

## 🔜 Roadmap

- [ ] Add support for more LLM models
- [ ] Implement user authentication
- [ ] Add collaborative features
- [ ] Improve code analysis capabilities
- [ ] Add support for more programming languages
- [ ] Implement caching for better performance

## 📞 Contact

Project Link: [https://github.com/Omersut/ghost-in-the-code](https://github.com/Omersut/ghost-in-the-code) # ghost-in-the-code
