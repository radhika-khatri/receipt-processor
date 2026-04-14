# 🧾 Receipt Processor

Upload a photo of a receipt and this app will automatically extract the items, quantities, and prices from it. It uses a vision AI model to read the receipt text and a language model to parse the items into structured data. You can then review, edit, or remove items before saving.

---

## 🚀 Live Demo

Run it locally by following the setup steps below. The app opens in your browser at `http://localhost:8000`.

---

## 🧠 How It Works

1. You upload one or more receipt images through the web interface.
2. The backend preprocesses each image (denoising, contrast enhancement, resizing) to improve OCR accuracy.
3. The enhanced image is passed to **Llama 3.2 Vision** (via Ollama) which extracts all visible text.
4. The raw text is then sent to **Mistral** (also via Ollama) which parses out individual items in a structured format.
5. Items are stored in memory and displayed in a combined summary table.
6. You can edit item names, quantities, and prices, or remove items you don't want.

---

## 📁 Repo Structure

```
receipt-processor/
├── main.py           # FastAPI backend with all processing logic
├── templates/        # HTML frontend (ind.html)
├── uploads/          # Temporary storage for uploaded images
└── .vscode/          # Editor config
```

---

## ⚙️ Requirements

- Python 3.x
- [Ollama](https://ollama.com/) installed and running locally
- The following Ollama models pulled:
  ```bash
  ollama pull llama3.2-vision
  ollama pull mistral
  ```
- Python dependencies:
  ```bash
  pip install fastapi uvicorn opencv-python pillow numpy ollama
  ```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/radhika-khatri/receipt-processor.git
cd receipt-processor
```

### 2. Start Ollama

Make sure Ollama is running in the background before starting the app:

```bash
ollama serve
```

### 3. Run the app

```bash
python main.py
```

The app will start at `http://localhost:8000`.

---

## 🖥️ How to Use It

1. Open `http://localhost:8000` in your browser.
2. Upload one or more receipt images (JPG or PNG).
3. The app processes each receipt and shows a summary table with extracted items, quantities, and prices.
4. Edit any item by clicking on it and updating the name, quantity, or price.
5. Remove unwanted items using the remove button.
6. Use **Save Value** to export the raw pixel data as CSV files for further analysis.
7. Use **Clear** to reset all data and start fresh.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main web interface |
| `POST` | `/upload` | Upload and process receipt images |
| `POST` | `/upload-progressive` | Upload with streamed real-time progress |
| `GET` | `/summary` | Get the current item summary |
| `POST` | `/update-item` | Edit an item's name, quantity, or price |
| `DELETE` | `/remove-item` | Remove a specific item |
| `DELETE` | `/clear` | Clear all receipts and items |
| `GET` | `/health` | Check backend and Ollama connection status |
| `GET` | `/debug` | Inspect internal state for debugging |

---

## ⚠️ Things to Keep in Mind

- Ollama must be running before you start the app. If it's not, the health check at `/health` will tell you.
- Image quality directly affects extraction accuracy. Blurry or poorly lit receipts may produce incomplete results.
- All data is stored in memory and resets when the server restarts. There is no persistent database.
- The app works best with printed receipts. Handwritten receipts may not extract cleanly.

---

## 🛠️ Built With

- **FastAPI** for the backend API
- **Ollama** with Llama 3.2 Vision and Mistral for AI processing
- **OpenCV** for image preprocessing
- **Pillow** and **NumPy** for image handling
- **Jinja2** for HTML templating

---

## 👤 Author

**Radhika Khatri**  
[GitHub Profile](https://github.com/radhika-khatri)
