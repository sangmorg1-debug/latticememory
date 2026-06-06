export function buildIndexText({ title, url, query }) {
  if (query && !title && !url) {
    return String(query);
  }
  return `${title || url || ""}\n${url || ""}`.trim();
}

export function hammingDistance(leftHex, rightHex, maxDistance = Number.POSITIVE_INFINITY) {
  if (!leftHex || !rightHex || leftHex.length !== rightHex.length) {
    return Number.POSITIVE_INFINITY;
  }
  let distance = 0;
  for (let i = 0; i < leftHex.length; i += 2) {
    if (leftHex.slice(i, i + 2) !== rightHex.slice(i, i + 2)) {
      distance += 1;
      if (distance > maxDistance) {
        return distance;
      }
    }
  }
  return distance;
}

export function searchEntriesByKey(entries, queryKey, maxHammingDistance) {
  return Array.from(entries || [])
    .map((entry) => ({
      ...entry,
      hammingDistance: hammingDistance(queryKey, entry.e8Key, maxHammingDistance)
    }))
    .filter((entry) => entry.hammingDistance <= maxHammingDistance)
    .sort((a, b) => b.visitedAt - a.visitedAt)
    .slice(0, 10);
}
