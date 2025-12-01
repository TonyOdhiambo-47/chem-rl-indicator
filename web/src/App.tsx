// src/App.tsx

import { useState, useMemo, useEffect } from "react";
import {
  computePHWeakAcidTitration,
  indicatorRgbFromPH,
  rgbToCss,
} from "./chem";
import EpisodeAnimation from "./EpisodeAnimation";
import "./App.css";

function App() {
  const [mode, setMode] = useState<"manual" | "animation">("manual");
  const [episodeData, setEpisodeData] = useState<any>(null);
  const [Va, setVa] = useState(50);       // mL
  const [Ca, setCa] = useState(0.1);      // M
  const [Cb, setCb] = useState(0.1);      // M
  const [pKa, setpKa] = useState(4.76);   // acetic acid
  const [Vb, setVb] = useState(0);        // base added (mL)

  // Load episode data on mount
  useEffect(() => {
    fetch("/episode_data.json")
      .then((res) => res.json())
      .then((data) => {
        setEpisodeData(data);
      })
      .catch(() => {
        // File doesn't exist, that's okay
        console.log("No episode_data.json found. Run export_episode.py first.");
      });
  }, []);

  const pH = useMemo(
    () => computePHWeakAcidTitration(Va, Ca, Vb, Cb, pKa),
    [Va, Ca, Vb, Cb, pKa]
  );

  const rgb = useMemo(
    () => indicatorRgbFromPH(pH, 7.0, 0.15),
    [pH]
  );

  const color = rgbToCss(rgb);

  // equivalence volume for UI reference:
  const Veq_mL = (Ca * Va) / Cb;

  // Show animation if episode data is loaded and in animation mode
  if (mode === "animation" && episodeData) {
    return (
      <div className="App">
        <div className="mode-switcher">
          <button
            onClick={() => setMode("manual")}
            className={mode === "manual" ? "active" : ""}
          >
            Manual Control
          </button>
          <button
            onClick={() => setMode("animation")}
            className={mode === "animation" ? "active" : ""}
          >
            Live Agent Animation
          </button>
        </div>
        <EpisodeAnimation episodeData={episodeData} speed={200} autoPlay={true} />
      </div>
    );
  }

  return (
    <div className="App">
      <div className="mode-switcher">
        <button
          onClick={() => setMode("manual")}
          className={mode === "manual" ? "active" : ""}
        >
          Manual Control
        </button>
        <button
          onClick={() => setMode("animation")}
          className={mode === "animation" ? "active" : ""}
        >
          Live Agent Animation
        </button>
      </div>
      <h1>Chem RL Indicator – Titration Visualizer</h1>
      <div className="controls">
        <div className="control-group">
          <label>
            Acid volume Va (mL):
            <input
              type="number"
              value={Va}
              step={1}
              onChange={(e) => setVa(Number(e.target.value))}
            />
          </label>
          <label>
            Acid concentration Ca (M):
            <input
              type="number"
              value={Ca}
              step={0.01}
              onChange={(e) => setCa(Number(e.target.value))}
            />
          </label>
          <label>
            Base concentration Cb (M):
            <input
              type="number"
              value={Cb}
              step={0.01}
              onChange={(e) => setCb(Number(e.target.value))}
            />
          </label>
          <label>
            Acid pKa:
            <input
              type="number"
              value={pKa}
              step={0.01}
              onChange={(e) => setpKa(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="slider-group">
          <label>
            Base volume added Vb (mL): {Vb.toFixed(2)} mL
          </label>
          <input
            type="range"
            min={0}
            max={Math.max(Veq_mL * 2, 1)}
            step={0.05}
            value={Vb}
            onChange={(e) => setVb(Number(e.target.value))}
          />
          <div className="veq-label">
            Equivalence volume ≈ {Veq_mL.toFixed(2)} mL
          </div>
        </div>
      </div>
      <div className="display">
        <div className="info-panel">
          <h2>Solution state</h2>
          <p>
            pH ≈ <strong>{pH.toFixed(2)}</strong>
          </p>
          <p>
            Vb / Veq ≈{" "}
            <strong>{(Vb / Veq_mL).toFixed(2)}</strong>
          </p>
          <p>
            Indicator color (simulated) – continuous with neutral band around 7.00
          </p>
        </div>
        <div className="color-panel">
          <div
            className="indicator-circle"
            style={{ backgroundColor: color }}
          />
          <p>{color}</p>
        </div>
      </div>
    </div>
  );
}

export default App;

