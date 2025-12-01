// src/chem.ts

export function computePHWeakAcidTitration(
  Va_mL: number,
  Ca: number,
  Vb_mL: number,
  Cb: number,
  pKa: number
): number {
  const Kw = 1e-14;
  const Ka = Math.pow(10, -pKa);
  const Va_L = Va_mL / 1000.0;
  const Vb_L = Vb_mL / 1000.0;
  const n_HA0 = Ca * Va_L;
  const n_OH = Cb * Vb_L;
  const Vtot = Va_L + Vb_L;

  if (Vtot <= 0) return 7.0;

  let pH: number;

  if (n_OH < n_HA0 - 1e-12) {
    // buffer region
    let n_A = n_OH;
    let n_HA = n_HA0 - n_OH;
    n_A = Math.max(n_A, 1e-16);
    n_HA = Math.max(n_HA, 1e-16);
    const ratio = n_A / n_HA;
    pH = pKa + Math.log10(ratio);
  } else if (Math.abs(n_OH - n_HA0) <= 1e-12) {
    // equivalence: A- only, weak base
    const C_A = n_HA0 / Vtot;
    const Kb = Kw / Ka;
    const OH = Math.sqrt(Math.max(Kb * C_A, 1e-20));
    const pOH = -Math.log10(OH);
    pH = 14 - pOH;
  } else {
    // excess strong base
    const n_excess = n_OH - n_HA0;
    const OH = Math.max(n_excess / Vtot, 1e-20);
    const pOH = -Math.log10(OH);
    pH = 14 - pOH;
  }

  return Math.min(Math.max(pH, 0), 14);
}

export function indicatorRgbFromPH(
  pH: number,
  pKa_ind: number = 7.0,
  neutralBand: number = 0.15
): [number, number, number] {
  const acid = [1.0, 1.0, 0.0];  // yellow
  const base = [0.0, 0.0, 1.0];  // blue
  const neutral = [0.0, 1.0, 0.0]; // green

  // fraction of base-colored form
  const f_base = 1.0 / (1.0 + Math.pow(10, pKa_ind - pH));
  const fb = Math.min(Math.max(f_base, 0), 1);
  const fa = 1 - fb;

  const baseMix: [number, number, number] = [
    fa * acid[0] + fb * base[0],
    fa * acid[1] + fb * base[1],
    fa * acid[2] + fb * base[2],
  ];

  const pHTarget = 7.0;
  const dist = Math.abs(pH - pHTarget);
  if (dist >= neutralBand) {
    return baseMix;
  }

  const w = 1.0 - dist / neutralBand; // 1 at center, 0 at edge
  const rgb: [number, number, number] = [
    (1 - w) * baseMix[0] + w * neutral[0],
    (1 - w) * baseMix[1] + w * neutral[1],
    (1 - w) * baseMix[2] + w * neutral[2],
  ];

  return rgb;
}

export function rgbToCss([r, g, b]: [number, number, number]): string {
  const R = Math.round(r * 255);
  const G = Math.round(g * 255);
  const B = Math.round(b * 255);
  return `rgb(${R}, ${G}, ${B})`;
}

