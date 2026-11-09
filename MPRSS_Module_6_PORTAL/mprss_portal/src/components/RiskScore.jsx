import { AlertTriangle, Shield, TrendingUp } from 'lucide-react';

const RiskScore = ({ score, isMalicious, anomalyAlert, confidence }) => {
  const getScoreColor = () => {
    if (score >= 67) return '#ef4444';
    if (score >= 34) return '#f59e0b';
    return '#10b981';
  };

  const getRiskLabel = () => {
    if (score >= 67) return 'MALICIOUS';
    if (score >= 34) return 'SUSPICIOUS';
    return 'SAFE';
  };

  const scoreColor = getScoreColor();
  const riskLabel = getRiskLabel();
  
  // Circle calculations
  const size = 280;
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const center = size / 2;

  return (
    <div className="risk-score-container">
      <div className="risk-score-card">
        <h3 className="section-title">Risk Assessment</h3>
        
        <div className="score-visual">
          <svg 
            width={size} 
            height={size} 
            viewBox={`0 0 ${size} ${size}`}
            className="circular-chart"
          >
            {/* Background circle */}
            <circle
              className="circle-bg"
              cx={center}
              cy={center}
              r={radius}
              stroke="#2a2e4a"
              strokeWidth={strokeWidth}
              fill="none"
            />
            {/* Progress circle */}
            <circle
              className="circle-progress"
              cx={center}
              cy={center}
              r={radius}
              stroke={scoreColor}
              strokeWidth={strokeWidth}
              fill="none"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              transform={`rotate(-90 ${center} ${center})`}
            />
            {/* Score number */}
            <text
              x={center}
              y={center - 8}
              textAnchor="middle"
              dominantBaseline="middle"
              className="score-text"
            >
              {score}
            </text>
            {/* Score label */}
            <text
              x={center}
              y={center + 30}
              textAnchor="middle"
              dominantBaseline="middle"
              className="score-label"
            >
              RISK SCORE
            </text>
          </svg>
        </div>

        <div className={`risk-badge ${riskLabel.toLowerCase()}`}>
          <Shield size={32} />
          <span>{riskLabel}</span>
        </div>

        {anomalyAlert && (
          <div className="anomaly-alert">
            <AlertTriangle size={22} />
            <span>ANOMALY ALERT: ZERO-DAY BEHAVIOR DETECTED</span>
          </div>
        )}

        <div className="score-metadata">
          <div className="metadata-item">
            <TrendingUp size={20} />
            <span>
              {confidence != null
                ? `Model Confidence: ${(confidence * 100).toFixed(0)}%`
                : 'Model Confidence: —'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskScore;