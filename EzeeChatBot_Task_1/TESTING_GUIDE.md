# Quick Start & Testing Guide — EzeeChatBot

Welcome! This guide will help you get the EzeeChatBot running on your machine and test it step by step — no coding knowledge needed for testing.

---

## What is this?

EzeeChatBot is a backend API that lets you:
1. **Upload any text or webpage URL** → it becomes a private knowledge base
2. **Ask questions** → the bot answers only from what you uploaded (no making stuff up)
3. **Check stats** → see how many questions were answered, costs, and response speed

---

## Part 1: Getting it Running

### Prerequisites
- Python 3.9 or newer installed on your machine
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Step 1 — Download the code
```bash
git clone https://github.com/SEK-11/ChatBOT_TASK1.git
cd ChatBOT_TASK1
```

### Step 2 — Create a virtual environment
This keeps all the packages isolated so they don't conflict with anything else on your computer.

```bash
python -m venv .venv
```

Then activate it:
- **Windows:** `.venv\Scripts\activate`
- **Mac / Linux:** `source .venv/bin/activate`

You'll see `(.venv)` appear at the start of your terminal line. That means it's active.

### Step 3 — Install packages
```bash
pip install -r requirements.txt
```
This will take 2–3 minutes the first time. It also automatically downloads the AI embedding model (~90 MB) and caches it.

### Step 4 — Add your OpenAI API Key
Rename `.env.example` to `.env`, then open it and replace the placeholder with your real key:
```
OPENAI_API_KEY=sk-your-real-key-here
```

### Step 5 — Start the server
```bash
uvicorn app.main:app --reload
```

You should see:
```
INFO: Application startup complete.
```
The API is now live at `http://127.0.0.1:8000`

---

## Part 2: Testing in the Browser (No Code Needed)

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

This is the **Swagger UI** — an interactive control panel where you can test every endpoint by clicking buttons and filling in forms.

---

### Test 1 — Upload a Document

1. Click the green **`POST /upload`** row to expand it
2. Click **"Try it out"** (top right of the section)
3. In the big text box, paste this sample:

```json
{
  "text": "The Amazon rainforest covers over 5.5 million square kilometres and produces 20 percent of the world's oxygen. It is home to 10 percent of all species on Earth. The Amazon River is the largest river by water discharge in the world. Deforestation remains one of the biggest threats to the Amazon, with cattle ranching and agriculture being the primary drivers.",
  "source_name": "amazon_facts"
}
```

4. Click the blue **"Execute"** button
5. Scroll down — you'll see a response like:

```json
{
  "bot_id": "abc123-xxxx-xxxx-xxxx",
  "chunks_stored": 1,
  "source": "text"
}
```

📌 **Copy the `bot_id` — you'll need it for the next steps.**

---

### Test 2 — Ask a Question That IS in the Document

1. Click **`POST /chat`** to expand it
2. Click **"Try it out"**
3. Paste this (replacing `YOUR_BOT_ID_HERE` with the id you copied):

```json
{
  "bot_id": "YOUR_BOT_ID_HERE",
  "user_message": "What percentage of the world's oxygen does the Amazon produce?",
  "conversation_history": []
}
```

4. Click **"Execute"**
5. ✅ You'll get a streamed answer grounded directly in your uploaded text:
```
data: {"token": "The Amazon produces 20 percent of the world's oxygen."}
data: [DONE]
```

---

### Test 3 — Try to Trick It (Hallucination Test)

This is the most important test — it verifies the bot won't make things up.

1. Stay on **`POST /chat`**
2. Change only the `user_message` to something NOT in your document:

```json
{
  "bot_id": "YOUR_BOT_ID_HERE",
  "user_message": "What is the population of Tokyo?",
  "conversation_history": []
}
```

3. Click **"Execute"**
4. ✅ The bot will **refuse to answer** instead of making something up:
```
data: {"token": "I couldn't find anything in the uploaded document that answers your question. Could you try rephrasing, or check if this topic is covered in the knowledge base?"}
data: [DONE]
```

This proves the bot is grounded — it won't invent facts outside its knowledge base.

---

### Test 4 — Check Statistics

After running a few chats, check your bot's analytics:

1. Click **`GET /stats/{bot_id}`** to expand it
2. Click **"Try it out"**
3. Paste your `bot_id` in the field
4. Click **"Execute"**
5. You'll see:

```json
{
  "bot_id": "YOUR_BOT_ID_HERE",
  "total_messages": 2,
  "avg_latency_ms": 850.5,
  "estimated_cost_usd": 0.000042,
  "unanswered_questions": 1
}
```

| Field | Meaning |
|---|---|
| `total_messages` | How many times `/chat` was called |
| `avg_latency_ms` | Average response time in milliseconds |
| `estimated_cost_usd` | How much the OpenAI calls cost so far |
| `unanswered_questions` | How many times the bot correctly refused to answer (hallucination blocks) |

---

### Test 5 — Upload a URL Instead of Text

You can also point the bot at a live webpage:

```json
{
  "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
  "source_name": "python_wiki"
}
```

Then ask questions about Python — the bot will fetch and index the entire article automatically.

---

## Part 3: Testing with curl (Optional for Technical Users)

If you prefer the command line over the browser UI:

**Upload:**
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here", "source_name": "my_doc"}'
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"bot_id": "YOUR_BOT_ID", "user_message": "Your question here", "conversation_history": []}'
```

**Stats:**
```bash
curl http://localhost:8000/stats/YOUR_BOT_ID
```

---

## Something Went Wrong?

| Problem | Fix |
|---|---|
| `pip install` fails on Windows | Make sure `.venv\Scripts\activate` was run first |
| "No knowledge base found" error | You used a wrong or expired `bot_id` — re-upload and copy the new one |
| Bot gives no OpenAI response | Check your `.env` file has a valid `OPENAI_API_KEY` |
| Port already in use | Another server is running on 8000. Stop it or run `uvicorn app.main:app --port 8001` |
