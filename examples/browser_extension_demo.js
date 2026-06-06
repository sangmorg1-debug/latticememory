/**
 * Phase 1b Flagship Product Demo: Local Browser Extension Search
 * 
 * Simulates local E8 lattice snapping, history indexing, and semantic matching
 * running entirely inside a sandboxed JavaScript environment (such as a browser extension).
 */

// --- E8 Snapping Math Ported to JavaScript ---

function buildShell1Codebook() {
    const codebook = [];
    // 1. Permutations of two non-zero coordinates: \pm e_i \pm e_j
    for (let i = 0; i < 8; i++) {
        for (let j = i + 1; j < 8; j++) {
            for (const si of [-1.0, 1.0]) {
                for (const sj of [-1.0, 1.0]) {
                    const v = new Array(8).fill(0.0);
                    v[i] = si;
                    v[j] = sj;
                    codebook.push(v);
                }
            }
        }
    }
    // 2. Half-integers with even sum
    for (let mask = 0; mask < 256; mask++) {
        const signs = new Array(8).fill(1.0);
        let negCount = 0;
        for (let bit = 0; bit < 8; bit++) {
            if ((mask >> bit) & 1) {
                signs[bit] = -1.0;
                negCount++;
            }
        }
        if (negCount % 2 === 0) {
            const v = [];
            for (let idx = 0; idx < 8; idx++) {
                v.push(signs[idx] * 0.5);
            }
            codebook.push(v);
        }
    }
    return codebook;
}

function decodeD8(x) {
    const z = new Array(8);
    let sumZ = 0;
    for (let i = 0; i < 8; i++) {
        const r = Math.round(x[i]);
        z[i] = r;
        sumZ += r;
    }
    // Correct JS modulo for negative numbers
    const parity = ((sumZ % 2) + 2) % 2;
    if (parity === 0) {
        return z;
    }
    
    let worstIdx = 0;
    let worstErr = -1.0;
    for (let i = 0; i < 8; i++) {
        const err = Math.abs(x[i] - z[i]);
        if (err > worstErr) {
            worstErr = err;
            worstIdx = i;
        }
    }
    
    const diff = x[worstIdx] - z[worstIdx];
    const adj = diff >= 0.0 ? 1.0 : -1.0;
    const zFixed = [...z];
    zFixed[worstIdx] += adj;
    return zFixed;
}

function e8Nearest(x) {
    const z0 = decodeD8(x);
    const xShift = x.map(val => val - 0.5);
    const z1Raw = decodeD8(xShift);
    const z1 = z1Raw.map(val => val + 0.5);
    
    let d0 = 0.0;
    let d1 = 0.0;
    for (let i = 0; i < 8; i++) {
        d0 += Math.pow(x[i] - z0[i], 2);
        d1 += Math.pow(x[i] - z1[i], 2);
    }
    return d0 <= d1 ? z0 : z1;
}

function snapEmbedding(embedding, codebook) {
    const dModel = embedding.length;
    const numBlocks = dModel / 8;
    const bytes = [];
    
    for (let b = 0; b < numBlocks; b++) {
        const block = embedding.slice(b * 8, (b + 1) * 8);
        
        let sum = 0.0;
        for (const x of block) sum += x * x;
        const norm = Math.sqrt(sum);
        
        const beta = (norm > 1e-8 ? norm : 1e-8) / Math.sqrt(2.0);
        const scaledBlock = block.map(val => val / beta);
        
        const snapped = e8Nearest(scaledBlock);
        
        let sumSnapped = 0.0;
        for (const x of snapped) sumSnapped += x * x;
        const snappedNorm = Math.sqrt(sumSnapped);
        const snClamped = snappedNorm > 1e-8 ? snappedNorm : 1e-8;
        const snappedUnit = snapped.map(val => val / snClamped);
        
        let maxDot = -999999.0;
        let bestIdx = 0;
        for (let k = 0; k < 240; k++) {
            let dot = 0.0;
            for (let i = 0; i < 8; i++) {
                dot += snappedUnit[i] * codebook[k][i];
            }
            if (dot > maxDot) {
                maxDot = dot;
                bestIdx = k;
            }
        }
        bytes.push(bestIdx);
    }
    return bytes;
}

// --- Browser Extension Indexing Simulation ---

// Generate dummy deterministic embeddings for mock pages (dimension = 32)
function getMockEmbedding(seed) {
    const vector = [];
    let state = seed;
    for (let i = 0; i < 32; i++) {
        // LCG random generator
        state = (state * 1103515245 + 12345) % 2147483648;
        vector.push((state / 2147483648) * 2.0 - 1.0);
    }
    // Normalize L2
    let sum = 0;
    for (const x of vector) sum += x * x;
    const norm = Math.sqrt(sum);
    return vector.map(x => x / norm);
}

function runBrowserExtensionDemo() {
    console.log("--- Phase 1b: Offline-First Privacy Browser Extension Demo ---");
    
    const codebook = buildShell1Codebook();
    
    // 1. Mock user browsing history
    const historyPages = [
        { url: "https://news.ycombinator.com", title: "Hacker News tech forum", seed: 42 },
        { url: "https://github.com", title: "Developer version control platform", seed: 100 },
        { url: "https://wikipedia.org", title: "Online free encyclopedia", seed: 2026 }
    ];
    
    // 2. Index local history using E8 snap keys
    console.log("\n[Extension] Local-first indexing user history...");
    const localLatticeIndex = new Map();
    
    for (const page of historyPages) {
        const emb = getMockEmbedding(page.seed);
        const snapKey = snapEmbedding(emb, codebook).join(","); // Hex/string key for JS Map
        
        if (!localLatticeIndex.has(snapKey)) {
            localLatticeIndex.set(snapKey, []);
        }
        localLatticeIndex.get(snapKey).push(page);
        console.log(`Indexed URL: ${page.url} -> E8 Key: [${snapKey}]`);
    }
    
    // 3. Perform a local semantic query matching
    // Search query matches Wikipedia seed (2026) exactly
    const searchQuery = "knowledge articles database";
    const queryEmb = getMockEmbedding(2026);
    const queryKey = snapEmbedding(queryEmb, codebook).join(",");
    
    console.log(`\n[Extension] User queries: "${searchQuery}"`);
    console.log(`[Extension] Query snapped to E8 Key: [${queryKey}]`);
    
    console.log("[Extension] Performing $O(1)$ address lookup...");
    const start = performance.now();
    const hits = localLatticeIndex.get(queryKey) || [];
    const end = performance.now();
    
    if (hits.length > 0) {
        console.log(`[Extension] Match found in ${(end - start).toFixed(4)} ms!`);
        for (const hit of hits) {
            console.log(`  -> URL: ${hit.url} ("${hit.title}")`);
        }
    } else {
        console.log("[Extension] Miss - No matching E8 keys in history.");
    }
    console.log("\nLocal search executed with 100% privacy on-device.");
}

runBrowserExtensionDemo();
