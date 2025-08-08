// index.js
// HackRx Level-4 webhook: LLM + semantic retrieval over uploaded documents
// Usage: POST /hackrx/run  { documents: ["https://...pdf", "...docx"], query: "46M, knee surgery, Pune, 3-month policy" }
// Env: OPENAI_API_KEY required
// Notes: designed to run on Replit/Glitch. Small, self-contained, in-memory vector store.

const express = require("express");
const fetch = require("node-fetch");
const pdf = require("pdf-parse");
const mammoth = require("mammoth");
const { OpenAI } = require("openai");

const app = express();
app.use(express.json({ limit: "20mb" }));

// Config
const OPENAI_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_KEY) {
  console.error("Missing OPENAI_API_KEY in environment variables.");
}
const openai = new OpenAI({ apiKey: OPENAI_KEY });

// Simple helpers
function cosine(a, b) {
  const dot = a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}

function chunkText(text, chunkSize = 800, overlap = 100) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const chunk = text.slice(i, i + chunkSize);
    chunks.push(chunk);
    i += chunkSize - overlap;
  }
  return chunks;
}

// Document fetch + extraction (PDF, DOCX, plain text, HTML fallback)
async function fetchAndExtract(urlOrData) {
  // urlOrData may be a URL string or an object {filename, content (base64)} - basic support
  try {
    if (typeof urlOrData === "object" && urlOrData.content) {
      // assume base64 text/pdf/docx
      const buffer = Buffer.from(urlOrData.content, "base64");
      if ((urlOrData.filename || "").toLowerCase().endsWith(".pdf")) {
        const data = await pdf(buffer);
        return { text: data.text, source: urlOrData.filename || "uploaded.pdf" };
      } else if ((urlOrData.filename || "").toLowerCase().endsWith(".docx")) {
        const res = await mammoth.extractRawText({ buffer });
        return { text: res.value, source: urlOrData.filename || "uploaded.docx" };
      } else {
        return { text: buffer.toString("utf8"), source: urlOrData.filename || "uploaded" };
      }
    }

    // assume URL
    const res = await fetch(urlOrData, { timeout: 30000 });
    if (!res.ok) {
      throw new Error(`Fetch failed ${res.status}`);
    }

    const contentType = res.headers.get("content-type") || "";
    const urlLower = urlOrData.toString().toLowerCase();

    if (contentType.includes("application/pdf") || urlLower.endsWith(".pdf")) {
      const buffer = await res.arrayBuffer();
      const data = await pdf(Buffer.from(buffer));
      return { text: data.text, source: urlOrData };
    } else if (contentType.includes("application/vnd.openxmlformats-officedocument.wordprocessingml.document") || urlLower.endsWith(".docx")) {
      const buffer = await res.arrayBuffer();
      const r = await mammoth.extractRawText({ buffer: Buffer.from(buffer) });
      return { text: r.value, source: urlOrData };
    } else {
      // plain text or HTML: try text
      const txt = await res.text();
      // If HTML, strip tags quickly (basic)
      const isHtml = /<\s*html|<\s*body|<\s*p|<\s*div/i.test(txt);
      const cleaned = isHtml ? txt.replace(/<[^>]*>/g, " ") : txt;
      return { text: cleaned, source: urlOrData };
    }
  } catch (err) {
    console.warn("fetchAndExtract error:", err.message);
    return { text: "", source: urlOrData, error: err.message };
  }
}

// Embed helper (OpenAI embeddings)
async function embedTexts(texts) {
  // texts: array of strings
  // returns: array of embedding vectors
  const batches = [];
  const results = [];
  // simple batching
  const BATCH = 10;
  for (let i = 0; i < texts.length; i += BATCH) {
    const slice = texts.slice(i, i + BATCH);
    const resp = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: slice
    });
    resp.data.forEach(d => results.push(d.embedding));
  }
  return results;
}

// Core endpoint
app.post("/hackrx/run", async (req, res) => {
  try {
    const body = req.body || {};
    // Accept documents as array of URLs or single URL or base64 objects
    let { documents, query } = body;
    if (!documents) {
      return res.status(400).json({ error: "Missing 'documents' (URL or array) in request body." });
    }
    if (!query) {
      return res.status(400).json({ error: "Missing 'query' in request body." });
    }

    if (!Array.isArray(documents)) documents = [documents];

    // 1) Fetch & extract all docs
    const docs = [];
    for (const item of documents) {
      const extracted = await fetchAndExtract(item);
      docs.push(extracted);
    }

    // 2) Split into chunks & prepare metadata
    const chunks = [];
    docs.forEach((doc, di) => {
      const txt = (doc.text || "") + "\n";
      const docChunks = chunkText(txt, 900, 150);
      docChunks.forEach((c, idx) =>
        chunks.push({
          text: c.replace(/\s+/g, " ").trim(),
          source: doc.source,
          doc_index: di,
          chunk_index: idx
        })
      );
    });

    if (chunks.length === 0) {
      return res.status(400).json({ error: "No extractable text found in documents." });
    }

    // 3) Embed chunks (caching could be added — in-memory for now)
    const texts = chunks.map(c => c.text);
    const embeddings = await embedTexts(texts);
    for (let i = 0; i < chunks.length; i++) chunks[i].embedding = embeddings[i];

    // 4) Embed query
    const qEmbResp = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: query
    });
    const qEmb = qEmbResp.data[0].embedding;

    // 5) Semantic search (top K)
    const K = Math.min(6, chunks.length);
    const scored = chunks.map(c => ({ ...c, score: cosine(qEmb, c.embedding) }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, K);

    // 6) Prepare context for LLM
    const contextConcat = top.map((t, i) => `-- Clause ${i+1} (source: ${t.source}, chunk_index:${t.chunk_index}, score:${t.score.toFixed(3)}) --\n${t.text}\n`).join("\n");

    const prompt = [
      { role: "system", content: "You are a helpful insurance policy analyst. Extract the rules/clause logic and decide whether the sample query should be approved, rejected, or require manual review. Return JSON exactly with keys: decision, amount, justification (array of {clause, source, chunk_index, similarity}), explain_prompt (human readable). Use values only; do not add extra commentary." },
      { role: "user", content: `Query: ${query}\n\nRelevant clauses (from documents):\n${contextConcat}\n\nTask:\n1) Parse the query into structured fields (age, sex, procedure, location, policy_duration_months).\n2) Evaluate each clause and determine the decision and amount (if any). If uncertain, choose "maybe" or "manual_review".\n3) Return the JSON object. Be explicit about which clauses you used (map to Clause 1..${top.length}).` }
    ];

    // 7) Call LLM
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini", // replace with available model you have access to
      messages: prompt,
      max_tokens: 600,
      temperature: 0.0
    });

    const llmText = completion.choices?.[0]?.message?.content || "";

    // Attempt to parse JSON from LLM output
    let parsed = null;
    try {
      // find first JSON block in the response
      const jsonMatch = llmText.match(/(\{[\s\S]*\})/);
      if (jsonMatch) parsed = JSON.parse(jsonMatch[1]);
    } catch (err) {
      // keep parsed null
    }

    // Build fallback response if parsing failed: create structured answer using best-effort heuristics
    if (!parsed) {
      // Heuristic: if any top chunk mentions "not covered" or "exclusion", reject; if mentions "covered" or procedure name -> approve
      const joined = top.map(t=>t.text.toLowerCase()).join(" ");
      const rejectKeywords = ["not covered","exclusion","deductible not","pre-existing","waiting period","not eligible","excludes"];
      const approveKeywords = ["covered","shall be paid","payable","benefit","eligible","entitled"];

      const foundReject = rejectKeywords.some(k => joined.includes(k));
      const foundApprove = approveKeywords.some(k => joined.includes(k));

      const decision = foundReject && !foundApprove ? "rejected" : (foundApprove && !foundReject ? "approved" : "maybe");
      parsed = {
        decision,
        amount: null,
        justification: top.map(t => ({ clause: t.text.slice(0,300), source: t.source, chunk_index: t.chunk_index, similarity: t.score })),
        explain_prompt: llmText || "LLM did not return parsable JSON; fallback heuristic used."
      };
    } else {
      // augment justification with similarity numbers from our top matches where we can map clause text
      if (Array.isArray(parsed.justification)) {
        parsed.justification = parsed.justification.map((j, i) => {
          // try match with top[i]
          const topMatch = top[i] || {};
          return {
            clause: j.clause || topMatch.text?.slice(0,300) || "",
            source: j.source || topMatch.source || "",
            chunk_index: j.chunk_index ?? topMatch.chunk_index ?? null,
            similarity: j.similarity ?? topMatch.score ?? null
          };
        });
      } else {
        parsed.justification = top.map(t => ({ clause: t.text.slice(0,300), source: t.source, chunk_index: t.chunk_index, similarity: t.score }));
      }
    }

    // 8) Final response
    return res.json({
      meta: {
        chunks_indexed: chunks.length,
        top_k: top.length,
        model: "embedding:text-embedding-3-small + gpt-4o-mini (chat)",
        timestamp: new Date().toISOString()
      },
      input: { query, documents },
      results: parsed
    });

  } catch (err) {
    console.error("Error in /hackrx/run:", err);
    return res.status(500).json({ error: err.message || "server error" });
  }
});

// Basic health
app.get("/", (req, res) => res.send("HackRx webhook is live. POST to /hackrx/run"));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Server listening on port", PORT));
