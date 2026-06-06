(() => {
  if (window.__latticeMemoryInjected) {
    return;
  }
  window.__latticeMemoryInjected = true;

  document.addEventListener("keydown", (event) => {
    if (!event.ctrlKey || !event.shiftKey || event.key.toLowerCase() !== "l") {
      return;
    }
    chrome.runtime.sendMessage(
      { type: "searchHistory", title: document.title, url: location.href },
      (response) => {
        if (chrome.runtime.lastError || !response?.results?.length) {
          return;
        }
        const top = response.results[0];
        if (top?.url) {
          location.href = top.url;
        }
      }
    );
  });
})();
