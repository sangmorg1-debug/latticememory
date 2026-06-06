const queryInput = document.getElementById("query");
const searchButton = document.getElementById("search");
const clearButton = document.getElementById("clear");
const exportButton = document.getElementById("export");
const countNode = document.getElementById("count");
const resultsNode = document.getElementById("results");

function send(type, payload = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, ...payload }, (response) => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError);
        return;
      }
      if (response?.error) {
        reject(new Error(response.error));
        return;
      }
      resolve(response);
    });
  });
}

function renderResults(results) {
  resultsNode.textContent = "";
  for (const result of results || []) {
    const link = document.createElement("a");
    link.className = "result";
    link.href = result.url;
    link.target = "_blank";
    const title = document.createElement("div");
    title.className = "result-title";
    title.textContent = result.title || result.url;
    const url = document.createElement("div");
    url.className = "result-url";
    url.textContent = result.url;
    link.append(title, url);
    resultsNode.append(link);
  }
  if (!resultsNode.childElementCount) {
    resultsNode.textContent = "No exact E8 address hits.";
  }
}

async function refreshStats() {
  const stats = await send("stats");
  countNode.textContent = String(stats.count || 0);
}

async function searchHistory() {
  const query = queryInput.value.trim();
  if (!query) {
    renderResults([]);
    return;
  }
  const response = await send("searchHistory", { query });
  renderResults(response.results || []);
}

async function clearIndex() {
  await send("clearIndex");
  renderResults([]);
  await refreshStats();
}

async function exportIndex() {
  const response = await send("exportIndex");
  const blob = new Blob([JSON.stringify(response.entries || [], null, 2)], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "latticememory-history-export.json";
  link.click();
  URL.revokeObjectURL(url);
}

searchButton.addEventListener("click", () => searchHistory().catch(console.error));
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    searchHistory().catch(console.error);
  }
});
clearButton.addEventListener("click", () => clearIndex().catch(console.error));
exportButton.addEventListener("click", () => exportIndex().catch(console.error));

refreshStats().catch(console.error);
