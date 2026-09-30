use std::slice;

/// Shift each HxWxC u8 row horizontally with zero fill.
/// Returns 0 on success; negative values indicate invalid arguments.
#[no_mangle]
pub unsafe extern "C" fn dvcr_shift_rows_zero_u8(
    src: *const u8,
    dst: *mut u8,
    height: usize,
    width: usize,
    channels: usize,
    shifts: *const i32,
) -> i32 {
    if src.is_null() || dst.is_null() || shifts.is_null() || height == 0 || width == 0 || channels == 0 {
        return -1;
    }
    let row_bytes = match width.checked_mul(channels) {
        Some(v) => v,
        None => return -2,
    };
    let total = match height.checked_mul(row_bytes) {
        Some(v) => v,
        None => return -2,
    };
    let src_slice = slice::from_raw_parts(src, total);
    let dst_slice = slice::from_raw_parts_mut(dst, total);
    let sh = slice::from_raw_parts(shifts, height);
    dst_slice.fill(0);

    for y in 0..height {
        let shift = sh[y] as isize;
        if shift >= width as isize || shift <= -(width as isize) {
            continue;
        }
        let src_row = y * row_bytes;
        let dst_row = y * row_bytes;
        if shift >= 0 {
            let px = shift as usize;
            let count_px = width - px;
            let count = count_px * channels;
            let src_start = src_row;
            let dst_start = dst_row + px * channels;
            dst_slice[dst_start..dst_start + count]
                .copy_from_slice(&src_slice[src_start..src_start + count]);
        } else {
            let px = (-shift) as usize;
            let count_px = width - px;
            let count = count_px * channels;
            let src_start = src_row + px * channels;
            let dst_start = dst_row;
            dst_slice[dst_start..dst_start + count]
                .copy_from_slice(&src_slice[src_start..src_start + count]);
        }
    }
    0
}

/// NumPy-compatible linear interpolation across multiple knot rows.
/// X coordinates are shared by every row. Calculations use f64 internally,
/// matching numpy.interp's effective precision before conversion to float32.
#[no_mangle]
pub unsafe extern "C" fn dvcr_interp_rows_f32(
    knots: *const f32,
    rows: usize,
    coarse: usize,
    xk: *const f32,
    x: *const f32,
    n: usize,
    out: *mut f32,
) -> i32 {
    if knots.is_null() || xk.is_null() || x.is_null() || out.is_null() || rows == 0 || coarse < 2 || n == 0 {
        return -1;
    }
    let k = slice::from_raw_parts(knots, rows * coarse);
    let xk = slice::from_raw_parts(xk, coarse);
    let x = slice::from_raw_parts(x, n);
    let out = slice::from_raw_parts_mut(out, rows * n);

    for row in 0..rows {
        let base = row * coarse;
        let out_base = row * n;
        let mut j = 0usize;
        for ix in 0..n {
            let xv = x[ix] as f64;
            if xv <= xk[0] as f64 {
                out[out_base + ix] = k[base];
                continue;
            }
            if xv >= xk[coarse - 1] as f64 {
                out[out_base + ix] = k[base + coarse - 1];
                continue;
            }
            while j + 1 < coarse - 1 && xv > xk[j + 1] as f64 {
                j += 1;
            }
            let x0 = xk[j] as f64;
            let x1 = xk[j + 1] as f64;
            let y0 = k[base + j] as f64;
            let y1 = k[base + j + 1] as f64;
            let t = if x1 != x0 { (xv - x0) / (x1 - x0) } else { 0.0 };
            out[out_base + ix] = (y0 + t * (y1 - y0)) as f32;
        }
    }
    0
}
