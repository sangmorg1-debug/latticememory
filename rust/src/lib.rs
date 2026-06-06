use wasm_bindgen::prelude::*;

fn build_shell1_codebook() -> Vec<[f32; 8]> {
    let mut codebook = Vec::with_capacity(240);
    // 1. Permutations of two non-zero coordinates: \pm e_i \pm e_j
    for i in 0..8 {
        for j in (i + 1)..8 {
            for si in &[-1.0, 1.0] {
                for sj in &[-1.0, 1.0] {
                    let mut v = [0.0; 8];
                    v[i] = *si;
                    v[j] = *sj;
                    codebook.push(v);
                }
            }
        }
    }
    // 2. Half-integers with even sum: 1/2 * (\pm 1, \dots, \pm 1)
    for mask in 0..256 {
        let mut signs = [1.0; 8];
        let mut neg_count = 0;
        for bit in 0..8 {
            if (mask >> bit) & 1 == 1 {
                signs[bit] = -1.0;
                neg_count += 1;
            }
        }
        if neg_count % 2 == 0 {
            let mut v = [0.0; 8];
            for idx in 0..8 {
                v[idx] = signs[idx] * 0.5;
            }
            codebook.push(v);
        }
    }
    codebook
}

fn decode_d8(x: &[f32; 8]) -> [f32; 8] {
    let mut z = [0.0; 8];
    let mut z_round = [0; 8];
    let mut sum_z = 0;
    for i in 0..8 {
        let r = x[i].round();
        z[i] = r;
        z_round[i] = r as i32;
        sum_z += z_round[i];
    }
    
    let parity = sum_z.rem_euclid(2);
    if parity == 0 {
        return z;
    }
    
    // Odd parity: need to adjust the coordinate that has the largest rounding error
    let mut worst_idx = 0;
    let mut worst_err = -1.0;
    for i in 0..8 {
        let err = (x[i] - z[i]).abs();
        if err > worst_err {
            worst_err = err;
            worst_idx = i;
        }
    }
    
    let diff = x[worst_idx] - z[worst_idx];
    let adj = if diff >= 0.0 { 1.0 } else { -1.0 };
    
    let mut z_fixed = z;
    z_fixed[worst_idx] += adj;
    z_fixed
}

fn e8_nearest(x: &[f32; 8]) -> [f32; 8] {
    let z0 = decode_d8(x);
    
    let mut x_shift = [0.0; 8];
    for i in 0..8 {
        x_shift[i] = x[i] - 0.5;
    }
    let z1_raw = decode_d8(&x_shift);
    let mut z1 = [0.0; 8];
    for i in 0..8 {
        z1[i] = z1_raw[i] + 0.5;
    }
    
    let mut d0 = 0.0;
    let mut d1 = 0.0;
    for i in 0..8 {
        d0 += (x[i] - z0[i]).powi(2);
        d1 += (x[i] - z1[i]).powi(2);
    }
    
    if d0 <= d1 {
        z0
    } else {
        z1
    }
}

fn l2_norm(v: &[f32; 8]) -> f32 {
    let mut sum = 0.0;
    for x in v {
        sum += x * x;
    }
    sum.sqrt()
}

fn dot_product(v1: &[f32; 8], v2: &[f32; 8]) -> f32 {
    let mut sum = 0.0;
    for i in 0..8 {
        sum += v1[i] * v2[i];
    }
    sum
}

fn snap_block(block: &[f32; 8], codebook: &[[f32; 8]; 240]) -> u8 {
    let norm = l2_norm(block);
    let beta = if norm > 1e-8 { norm } else { 1e-8 } / 2.0f32.sqrt();
    
    let mut scaled_block = [0.0; 8];
    for i in 0..8 {
        scaled_block[i] = block[i] / beta;
    }
    
    let snapped = e8_nearest(&scaled_block);
    let snapped_norm = l2_norm(&snapped);
    let sn_norm_clamped = if snapped_norm > 1e-8 { snapped_norm } else { 1e-8 };
    
    let mut snapped_unit = [0.0; 8];
    for i in 0..8 {
        snapped_unit[i] = snapped[i] / sn_norm_clamped;
    }
    
    let mut max_dot = -999999.0f32;
    let mut best_idx = 0;
    for k in 0..240 {
        let dot = dot_product(&snapped_unit, &codebook[k]);
        if dot > max_dot {
            max_dot = dot;
            best_idx = k;
        }
    }
    best_idx as u8
}

/// Snaps flat float32 embeddings to E8 lattice Shell 1 byte indices.
/// Returns a byte array where each block of 8 dimensions is represented by 1 byte.
#[wasm_bindgen]
pub fn snap_embeddings(embeddings: &[f32], d_model: usize) -> Vec<u8> {
    if d_model == 0 || d_model % 8 != 0 || embeddings.is_empty() {
        return Vec::new();
    }
    let codebook_vec = build_shell1_codebook();
    let mut codebook = [[0.0; 8]; 240];
    for i in 0..240 {
        codebook[i] = codebook_vec[i];
    }
    
    let total_blocks = embeddings.len() / 8;
    let mut result = Vec::with_capacity(total_blocks);
    
    let mut block = [0.0; 8];
    for chunk in embeddings.chunks_exact(8) {
        block.copy_from_slice(chunk);
        result.push(snap_block(&block, &codebook));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_size() {
        let codebook = build_shell1_codebook();
        assert_eq!(codebook.len(), 240);
    }

    #[test]
    fn test_snapping_deterministic() {
        let mut emb = vec![0.0; 384];
        emb[0] = 1.0;
        emb[1] = -0.5;
        let keys1 = snap_embeddings(&emb, 384);
        let keys2 = snap_embeddings(&emb, 384);
        assert_eq!(keys1, keys2);
        assert_eq!(keys1.len(), 48); // 384 / 8 = 48 bytes
    }
}
