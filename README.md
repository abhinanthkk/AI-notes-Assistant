# AI Notes Q&A Assistant

A production-ready Python web application for semantic search and question-answering over PDF documents. Upload your PDF notes, and ask questions to retrieve the most relevant text chunks with page numbers using embeddings and FAISS vector search.

## 🎯 Features

- **PDF Upload**: Upload one or more PDF files for processing
- **Text Extraction**: Extract text page-wise from PDFs using pdfplumber
- **Smart Chunking**: Chunk text with configurable overlap (default: 500 words, 100 word overlap)
- **Semantic Embeddings**: Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: Fast similarity search using FAISS CPU
- **Persistent Storage**: Save and reload indexes automatically
- **Multi-PDF Support**: Search across multiple PDFs in a single workspace
- **Rich Results**: Display relevant text snippets with PDF name, page number, and similarity scores

## 🛠️ Tech Stack

- **Streamlit**: Web UI framework
- **sentence-transformers**: Embedding generation
- **FAISS CPU**: Vector similarity search
- **pdfplumber**: PDF text extraction
- **numpy**: Numerical operations

## 📁 Project Structure

```
ai-notes-qa-app/
├── app/
│   ├── main.py           # Streamlit UI application
│   ├── pdf_reader.py     # PDF text extraction
│   ├── chunker.py        # Text chunking with overlap
│   ├── embeddings.py     # Embedding generation
│   ├── retriever.py      # FAISS-based retrieval
│   ├── storage.py        # Persistent storage management
│   └── utils.py          # Utility functions
├── storage/
│   ├── pdfs/             # Stored PDF files
│   ├── indexes/          # FAISS indexes
│   └── embeddings/       # Embedding cache (if needed)
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
└── .gitignore           # Git ignore rules
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- python3-venv package (for creating virtual environments)

### Install python3-venv (if needed)

On Debian/Ubuntu systems, you may need to install the venv package first:

```bash
sudo apt install python3-venv
```

### Steps

1. **Clone or download this repository**

2. **Create a virtual environment (required)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
   You should see `(venv)` in your terminal prompt after activation.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

The app will automatically open in your browser at `http://localhost:8501`.

### Note on Virtual Environments

**Always activate the virtual environment before running the app:**
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

If you see `(venv)` in your terminal prompt, you're in the virtual environment.

## 📖 How to Use

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Process & Index**: Click "Process & Index PDFs" to extract text, generate embeddings, and build the search index
3. **Ask Questions**: Enter your question in the search box and click "Search"
4. **View Results**: Review the most relevant text chunks with:
   - PDF filename
   - Page number
   - Similarity score
   - Relevant text snippet

## 🔬 How It Works

### Embeddings & FAISS Retrieval

1. **Text Extraction**: PDFs are processed page-by-page to extract text content
2. **Chunking**: Text is split into overlapping chunks (~500 words, 100 word overlap) to maintain context
3. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model
4. **Index Building**: Embeddings are stored in a FAISS index for fast similarity search
5. **Query Processing**: User queries are converted to embeddings and searched against the index
6. **Result Ranking**: Top-k most similar chunks are returned based on cosine similarity (inner product on normalized vectors)

### Storage & Persistence

- PDFs are saved to `storage/pdfs/`
- FAISS indexes are saved to `storage/indexes/`
- Metadata (chunk info, page numbers) is stored as JSON
- The app automatically reloads existing indexes on startup

## 🚢 Deployment to GitHub

### Initial Setup

1. **Initialize Git repository** (if not already done)
   ```bash
   git init
   ```

2. **Add all files**
   ```bash
   git add .
   ```

3. **Commit**
   ```bash
   git commit -m "Initial commit: AI Notes Q&A Assistant"
   ```

4. **Create a new repository on GitHub**
   - Go to GitHub and create a new repository
   - Do NOT initialize with README, .gitignore, or license (we already have these)

5. **Connect and push**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Notes for GitHub

- The `storage/` directory is in `.gitignore` to avoid committing user data
- PDF files are excluded from version control
- Only source code and configuration files are tracked

## 🧪 Testing

The app includes error handling for:
- Empty PDFs
- Unreadable PDFs
- Missing indexes
- Invalid queries

On first run, the app will:
- Create storage directories automatically
- Load existing indexes if available
- Handle missing data gracefully

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using Python, Streamlit, and FAISS**

