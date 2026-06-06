import initWasm, { snap_embeddings } from "./wasm/latticememory_kernel.js";
import { buildIndexText, searchEntriesByKey } from "./search_core.js";

const DB_NAME = "latticememory-history";
const DB_VERSION = 1;
const STORE_NAME = "entries";
const D_MODEL = 384;
const MAX_HAMMING_DISTANCE = 1;

let wasmReady = null;

function ensureWasm() {
  if (!wasmReady) {
    const wasmUrl = chrome.runtime.getURL("wasm/latticememory_kernel_bg.wasm");
    wasmReady = initWasm(wasmUrl);
  }
  return wasmReady;
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      const store = db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
      store.createIndex("e8Key", "e8Key", { unique: false });
      store.createIndex("url", "url", { unique: false });
      store.createIndex("visitedAt", "visitedAt", { unique: false });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function withStore(mode, callback) {
  return openDatabase().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, mode);
    const store = tx.objectStore(STORE_NAME);
    const result = callback(store);
    tx.oncomplete = () => {
      db.close();
      resolve(result);
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  }));
}

function mockEncode(text) {
  const vector = new Float32Array(D_MODEL);
  let state = 2166136261;
  for (const ch of String(text)) {
    state ^= ch.charCodeAt(0);
    state = Math.imul(state, 16777619) >>> 0;
  }
  let norm = 0;
  for (let i = 0; i < D_MODEL; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const value = (state / 0xffffffff) * 2 - 1;
    vector[i] = value;
    norm += value * value;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < D_MODEL; i += 1) {
    vector[i] /= norm;
  }
  return vector;
}

async function snapText(text) {
  await ensureWasm();
  const embedding = mockEncode(text);
  const keyBytes = snap_embeddings(embedding, D_MODEL);
  return Array.from(keyBytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function indexPage({ url, title, lastVisitTime }) {
  if (!url || url.startsWith("chrome://") || url.startsWith("edge://")) {
    return;
  }
  const indexText = buildIndexText({ title, url });
  const e8Key = await snapText(indexText);
  const entry = {
    url,
    title: title || url,
    indexText,
    e8Key,
    visitedAt: lastVisitTime || Date.now()
  };
  await withStore("readwrite", (store) => store.put(entry));
}

async function handleSearch(message) {
  const searchText = buildIndexText({
    title: message.title,
    url: message.url,
    query: message.query || ""
  });
  const e8Key = await snapText(searchText);
  return withStore("readonly", (store) => new Promise((resolve, reject) => {
    const request = store.getAll();
    request.onsuccess = () => {
      const results = searchEntriesByKey(request.result, e8Key, MAX_HAMMING_DISTANCE);
      resolve({ e8Key, results });
    };
    request.onerror = () => reject(request.error);
  }));
}

async function handleStats() {
  return withStore("readonly", (store) => new Promise((resolve, reject) => {
    const request = store.count();
    request.onsuccess = () => resolve({ count: request.result, dModel: D_MODEL });
    request.onerror = () => reject(request.error);
  }));
}

async function handleClearIndex() {
  await withStore("readwrite", (store) => store.clear());
  return { cleared: true };
}

async function handleExportIndex() {
  return withStore("readonly", (store) => new Promise((resolve, reject) => {
    const request = store.getAll();
    request.onsuccess = () => resolve({ entries: request.result });
    request.onerror = () => reject(request.error);
  }));
}

chrome.history.onVisited.addListener((historyItem) => {
  indexPage(historyItem).catch((error) => {
    console.warn("LatticeMemory index failed", error);
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const run = async () => {
    if (message?.type === "searchHistory") {
      return handleSearch(message);
    }
    if (message?.type === "stats") {
      return handleStats();
    }
    if (message?.type === "clearIndex") {
      return handleClearIndex();
    }
    if (message?.type === "exportIndex") {
      return handleExportIndex();
    }
    return { error: "unknown message type" };
  };

  run().then(sendResponse).catch((error) => {
    sendResponse({ error: String(error?.message || error) });
  });
  return true;
});
