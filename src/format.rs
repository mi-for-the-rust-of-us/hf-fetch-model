// SPDX-License-Identifier: MIT OR Apache-2.0

//! Display formatters shared across binary subcommands.
//!
//! Currently exports [`format_size`] for human-readable byte counts. Future
//! formatters that are shared across more than one CLI subcommand (age,
//! short-`SHA`, parameter counts) belong here too.

/// Formats a byte count as a human-readable string with binary `IEC` units.
///
/// Buckets, by the value as it is displayed, after rounding:
///
/// | Range | Format |
/// |-------|--------|
/// | `< 1024` bytes | `"{N} B"` |
/// | below `1024.0 KiB` | `"{X.X} KiB"` |
/// | below `1000.00 MiB` | `"{X.XX} MiB"` |
/// | below `1000.00 GiB` | `"{X.XX} GiB"` |
/// | below `1000.00 TiB` | `"{X.XX} TiB"` |
/// | below `1000.00 PiB` | `"{X.XX} PiB"` |
/// | otherwise | `"{X.XX} EiB"` |
///
/// The unit is chosen *after* rounding: a count just below a boundary, which
/// would round to `1024.0 KiB` or `1000.00 MiB`, prints in the next unit
/// (`1.00 MiB`, `0.98 GiB`) instead. The `1000` thresholds above KiB, rather
/// than `1024`, keep the integer part to at most three digits, the CLI
/// convention since v0.9.3: a 1.0 GiB file prints as `"1.00 GiB"`, never
/// `"1024.00 MiB"`. So every `u64` prints in at most 10 characters (the
/// widest are `"1023.9 KiB"` and `"999.99 MiB"` and their peers, and
/// `u64::MAX` is `"16.00 EiB"`), which is what fixed-width size columns rely
/// on.
#[must_use]
pub fn format_size(bytes: u64) -> String {
    // Each unit below EiB, with its decimals and the displayed value at
    // which the next unit takes over.
    const STEPS: [(&str, usize, f64); 5] = [
        ("KiB", 1, 1024.0),
        ("MiB", 2, 1000.0),
        ("GiB", 2, 1000.0),
        ("TiB", 2, 1000.0),
        ("PiB", 2, 1000.0),
    ];

    if bytes < 1024 {
        return format!("{bytes} B");
    }
    // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let mut val = bytes as f64 / 1024.0;
    for (unit, decimals, next_at) in STEPS {
        let shown = format!("{val:.decimals$}");
        // Decided on the rounded string itself, so it cannot disagree with
        // what is printed.
        if !shown.parse::<f64>().is_ok_and(|v| v >= next_at) {
            return format!("{shown} {unit}");
        }
        val /= 1024.0;
    }
    format!("{val:.2} EiB")
}

#[cfg(test)]
mod tests {
    use super::format_size;

    #[test]
    fn bytes_under_kib() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1), "1 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn kib_range() {
        assert_eq!(format_size(1024), "1.0 KiB");
        assert_eq!(format_size(1536), "1.5 KiB");
    }

    #[test]
    fn mib_range() {
        assert_eq!(format_size(1024 * 1024), "1.00 MiB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.00 MiB");
    }

    #[test]
    fn gib_range_kicks_in_at_1000_mib() {
        // 999 MiB still formats as MiB; 1000 MiB flips to GiB.
        assert_eq!(format_size(999 * 1024 * 1024), "999.00 MiB");
        let gib = 1u64 << 30;
        assert_eq!(format_size(gib), "1.00 GiB");
    }

    #[test]
    fn tib_range_kicks_in_at_1000_gib() {
        let gib = 1u64 << 30;
        assert_eq!(format_size(999 * gib), "999.00 GiB");
        let tib = 1024 * gib;
        assert_eq!(format_size(tib), "1.00 TiB");
    }

    #[test]
    fn unit_is_chosen_after_rounding() {
        // The last count each unit still shows, then the first that rounds
        // up to the next unit's threshold and so flips. Before the fix these
        // printed `1024.0 KiB`, `1000.00 MiB` and `1000.00 GiB`.
        assert_eq!(format_size(1_048_524), "1023.9 KiB");
        assert_eq!(format_size(1_048_525), "1.00 MiB");
        assert_eq!(format_size((1 << 20) - 1), "1.00 MiB");
        assert_eq!(format_size(1_048_570_757), "999.99 MiB");
        assert_eq!(format_size(1_048_570_758), "0.98 GiB");
        assert_eq!(format_size(1000 * (1 << 20) - 1), "0.98 GiB");
        assert_eq!(format_size(1_073_736_455_290), "999.99 GiB");
        assert_eq!(format_size(1_073_736_455_291), "0.98 TiB");
        assert_eq!(format_size(1000 * (1 << 30) - 1), "0.98 TiB");
    }

    #[test]
    fn units_above_tib() {
        // TiB used to be the top unit, so a count from 1000 TiB up printed
        // an ever longer figure; `u64::MAX` printed `16777216.00 TiB`.
        assert_eq!(format_size(1_099_506_130_217_861), "999.99 TiB");
        assert_eq!(format_size(1000 << 40), "0.98 PiB");
        assert_eq!(format_size(1 << 50), "1.00 PiB");
        assert_eq!(format_size(1000 << 50), "0.98 EiB");
        assert_eq!(format_size(1 << 60), "1.00 EiB");
        assert_eq!(format_size(u64::MAX), "16.00 EiB");
    }

    #[test]
    fn output_never_exceeds_ten_characters() {
        // Every power of two and its neighbours, and every unit's `999`,
        // `1000`, `1023` and `1024` multiples and their neighbours: the
        // values where a unit boundary or a rounding carry can widen the
        // output. None may exceed 10 characters, and none may show a figure
        // that should have flipped to the next unit.
        let mut samples: Vec<u64> = Vec::new();
        for k in 0..64 {
            let p = 1u64 << k;
            samples.extend([p - 1, p, p + 1]);
        }
        samples.push(u64::MAX);
        for k in 0..6 {
            let unit = 1u64 << (10 * k);
            for n in [999u64, 1000, 1023, 1024] {
                if let Some(v) = n.checked_mul(unit) {
                    samples.extend([v.saturating_sub(1), v, v.saturating_add(1)]);
                }
            }
        }
        for bytes in samples {
            let s = format_size(bytes);
            assert!(
                s.len() <= 10,
                "{bytes} printed as {s:?}, over 10 characters"
            );
            assert!(
                !s.starts_with("1024.0 ") && !s.starts_with("1000.00 "),
                "{bytes} printed as {s:?}, which should have flipped to the next unit"
            );
        }
    }
}
