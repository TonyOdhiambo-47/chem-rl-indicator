// Live animation component for RL agent episodes
import { useState, useEffect, useRef } from "react";
import { computePHWeakAcidTitration, indicatorRgbFromPH, rgbToCss } from "./chem";
import "./EpisodeAnimation.css";

interface EpisodeStep {
  step: number;
  action: number;
  action_name: string;
  Vb_ml: number;
  pH: number;
  color: [number, number, number];
  reward: number;
  total_reward: number;
  distance_to_target: number;
  V_over_Veq: number;
  terminated: boolean;
  truncated: boolean;
}

interface EpisodeData {
  steps: EpisodeStep[];
  initial_state: {
    Vb_ml: number;
    pH: number;
    color: [number, number, number];
  };
  target_pH: number;
  Veq_ml: number;
  step_sizes_ml: number[];
  action_names: string[];
  summary: {
    total_steps: number;
    final_pH: number;
    final_Vb_ml: number;
    total_reward: number;
    final_distance: number;
    success: boolean;
  };
}

interface EpisodeAnimationProps {
  episodeData: EpisodeData | null;
  speed?: number; // milliseconds per step
  autoPlay?: boolean;
}

export default function EpisodeAnimation({
  episodeData,
  speed = 200,
  autoPlay = true,
}: EpisodeAnimationProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Reset when episode data changes
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(autoPlay);
  }, [episodeData, autoPlay]);

  // Auto-play animation
  useEffect(() => {
    if (!episodeData || !isPlaying) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    if (currentStep >= episodeData.steps.length - 1) {
      setIsPlaying(false);
      return;
    }

    intervalRef.current = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= episodeData.steps.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, speed);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, currentStep, episodeData, speed]);

  if (!episodeData) {
    return (
      <div className="episode-animation-container">
        <div className="no-data">
          <p>No episode data loaded</p>
          <p className="hint">Run: <code>python export_episode.py --model models/ppo_weak_acid_indicator.zip</code></p>
        </div>
      </div>
    );
  }

  const currentStepData = episodeData.steps[currentStep];
  const displayedSteps = showTrajectory
    ? episodeData.steps.slice(0, currentStep + 1)
    : [episodeData.steps[currentStep]];

  const progress = ((currentStep + 1) / episodeData.steps.length) * 100;

  return (
    <div className="episode-animation-container">
      <div className="animation-header">
        <h2>Live Agent Performance</h2>
        <div className="controls">
          <button onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
          <button onClick={() => setCurrentStep(0)}>⏮ Reset</button>
          <button onClick={() => setCurrentStep(episodeData.steps.length - 1)}>
            ⏭ End
          </button>
          <label>
            <input
              type="checkbox"
              checked={showTrajectory}
              onChange={(e) => setShowTrajectory(e.target.checked)}
            />
            Show trajectory
          </label>
        </div>
      </div>

      <div className="animation-content">
        {/* Left: Titration Curve */}
        <div className="curve-panel">
          <div className="curve-container">
            <svg
              viewBox="0 0 600 400"
              className="titration-svg"
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Grid lines */}
              <defs>
                <pattern
                  id="grid"
                  width="60"
                  height="28.57"
                  patternUnits="userSpaceOnUse"
                >
                  <path
                    d="M 60 0 L 0 0 0 28.57"
                    fill="none"
                    stroke="#333"
                    strokeWidth="0.5"
                    opacity="0.2"
                  />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />

              {/* pH 7 target line */}
              <line
                x1="0"
                y1="200"
                x2="600"
                y2="200"
                stroke="#666"
                strokeWidth="2"
                strokeDasharray="5,5"
                opacity="0.7"
              />
              {/* Move target label slightly up and right to avoid overlapping y-axis label */}
              <text x="40" y="180" fill="#666" fontSize="12">
                pH 7.0 (Target)
              </text>

              {/* Equivalence line */}
              <line
                x1={(episodeData.Veq_ml / 100) * 600}
                y1="0"
                x2={(episodeData.Veq_ml / 100) * 600}
                y2="400"
                stroke="#ff9500"
                strokeWidth="2"
                strokeDasharray="3,3"
                opacity="0.5"
              />

              {/* Trajectory path */}
              {displayedSteps.length > 1 && (
                <path
                  d={displayedSteps
                    .map(
                      (step, i) =>
                        `${i === 0 ? "M" : "L"} ${
                          (step.Vb_ml / 100) * 600
                        } ${400 - (step.pH / 14) * 400}`
                    )
                    .join(" ")}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="3"
                  className="trajectory-path"
                />
              )}

              {/* Data points */}
              {displayedSteps.map((step, i) => (
                <circle
                  key={i}
                  cx={(step.Vb_ml / 100) * 600}
                  cy={400 - (step.pH / 14) * 400}
                  r={i === displayedSteps.length - 1 ? 6 : 4}
                  fill={i === displayedSteps.length - 1 ? "#ef4444" : "#3b82f6"}
                  stroke="white"
                  strokeWidth={i === displayedSteps.length - 1 ? 2 : 1}
                  className="data-point"
                />
              ))}

              {/* Start marker */}
              <circle
                cx={(episodeData.initial_state.Vb_ml / 100) * 600}
                cy={400 - (episodeData.initial_state.pH / 14) * 400}
                r="8"
                fill="#10b981"
                stroke="white"
                strokeWidth="2"
              />
              <text
                x={(episodeData.initial_state.Vb_ml / 100) * 600}
                y={400 - (episodeData.initial_state.pH / 14) * 400 - 15}
                fill="#10b981"
                fontSize="10"
                textAnchor="middle"
                fontWeight="bold"
              >
                Start
              </text>

              {/* Current point label */}
              {currentStepData && (
                <g>
                  <text
                    x={(currentStepData.Vb_ml / 100) * 600 + 10}
                    y={400 - (currentStepData.pH / 14) * 400 - 10}
                    fill="#ef4444"
                    fontSize="11"
                    fontWeight="bold"
                  >
                    Step {currentStep + 1}
                  </text>
                </g>
              )}

              {/* Axes labels */}
              <text
                x="300"
                y="390"
                fill="#e5e7eb"
                fontSize="14"
                textAnchor="middle"
                fontWeight="bold"
              >
                Base Volume Added (mL)
              </text>
              <text
                x="30"
                y="210"
                fill="#e5e7eb"
                fontSize="14"
                textAnchor="middle"
                fontWeight="bold"
                transform="rotate(-90 30 210)"
              >
                pH
              </text>
            </svg>
          </div>
          
          {/* Action and Rewards at bottom of graph */}
          <div className="graph-bottom-info">
            <div className="graph-info-section">
              <h3>Action</h3>
              <div className="action-display">
                <span className="action-name">{currentStepData.action_name}</span>
                <span className="action-number">(Action {currentStepData.action})</span>
              </div>
            </div>
            
            <div className="graph-info-section">
              <h3>Rewards</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span className="label">Step Reward:</span>
                  <span className="value">{currentStepData.reward.toFixed(2)}</span>
                </div>
                <div className="info-item">
                  <span className="label">Total Reward:</span>
                  <span className="value highlight">
                    {currentStepData.total_reward.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Current State */}
        <div className="state-panel">
          <div className="indicator-display">
            <div
              className="indicator-circle-large"
              style={{
                backgroundColor: rgbToCss(currentStepData.color),
                boxShadow: `0 0 30px ${rgbToCss(currentStepData.color)}40`,
              }}
            />
            <p className="color-label">{rgbToCss(currentStepData.color)}</p>
          </div>

          <div className="state-info">
            <div className="info-section">
              <h3>Current State</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span className="label">pH:</span>
                  <span className="value">{currentStepData.pH.toFixed(2)}</span>
                </div>
                <div className="info-item">
                  <span className="label">Base Added:</span>
                  <span className="value">
                    {currentStepData.Vb_ml.toFixed(2)} mL
                  </span>
                </div>
                <div className="info-item">
                  <span className="label">Distance to Target:</span>
                  <span className="value">
                    {currentStepData.distance_to_target.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>

            <div className="progress-section">
              <div className="progress-bar-container">
                <div
                  className="progress-bar"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="progress-text">
                Step {currentStep + 1} / {episodeData.steps.length} (
                {progress.toFixed(1)}%)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

