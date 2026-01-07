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

## 🚢 Deployment

### Deploy to Streamlit Cloud (Recommended)

Streamlit Cloud offers free hosting for Streamlit apps directly from your GitHub repository.

#### Prerequisites
- Your code must be pushed to a GitHub repository
- You need a GitHub account

#### Steps

1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `abhinanthkk/AI-notes-Assistant`
   - Select branch: `main`
   - Main file path: `app/main.py`
   - Click "Deploy"

4. **Wait for deployment**
   - Streamlit Cloud will install dependencies from `requirements.txt`
   - First deployment may take 5-10 minutes
   - You'll get a public URL like: `https://your-app-name.streamlit.app`

#### Important Notes for Streamlit Cloud

- **Storage**: The `storage/` directory is ephemeral on Streamlit Cloud. Data may be lost when the app restarts. For production, consider using cloud storage (S3, Google Cloud Storage, etc.)
- **File Uploads**: Users can upload PDFs, but they won't persist across app restarts
- **Memory**: Free tier has memory limits. Large PDFs may cause issues
- **Auto-deploy**: Every push to the main branch automatically redeploys the app

### Local Deployment

For local deployment, simply run:
```bash
streamlit run app/main.py
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

