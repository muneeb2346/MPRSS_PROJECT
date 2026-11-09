import { Activity, FileCode, Brain, BarChart3 } from 'lucide-react';

const RiskBreakdown = ({ scores, combinedRisk }) => {
  const breakdownItems = [
    { 
      name: 'Static Score', 
      value: scores.static, 
      icon: FileCode,
      color: '#4f46e5',
      description: 'Permission & code analysis'
    },
    { 
      name: 'Behavioral Score', 
      value: scores.behavioral, 
      icon: Activity,
      color: '#8b5cf6',
      description: 'Runtime behavior analysis'
    },
    { 
      name: 'Anomaly Score', 
      value: scores.anomaly, 
      icon: Brain,
      color: '#ec4899',
      description: 'Zero-day pattern detection'
    },
    { 
      name: 'Combined Risk', 
      value: scores.mlConfidence, 
      icon: BarChart3,
      color: '#10b981',
      description: 'Module 6: weighted static + anomaly + behavioral score'
    }
  ];

  return (
    <div className="risk-breakdown">
      <h3 className="section-title">Risk Breakdown</h3>
      <div className="breakdown-grid">
        {breakdownItems.map((item, index) => {
          const Icon = item.icon;
          const isPending = item.value == null;
          return (
            <div
              key={item.name}
              className={`breakdown-card ${isPending ? 'pending' : ''}`}
              style={{ animationDelay: `${0.1 + index * 0.05}s`, opacity: isPending ? 0.55 : 1 }}
            >
              <div className="breakdown-header">
                <div className="icon-wrapper" style={{ background: `${item.color}20` }}>
                  <Icon size={30} color={item.color} />
                </div>
                <span className="breakdown-name">{item.name}</span>
              </div>
              <div className="breakdown-value">{isPending ? 'Pending' : `${item.value}%`}</div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: isPending ? '0%' : `${item.value}%`,
                    background: `linear-gradient(90deg, ${item.color}, ${item.color}dd)`
                  }}
                />
              </div>
              <p className="breakdown-desc">
                {isPending ? 'Module not yet integrated' : item.description}
              </p>
            </div>
          );
        })}
      </div>

      {combinedRisk && combinedRisk.explanation && (
        <div className="combined-risk-footer">
          <p className="combined-risk-explanation">{combinedRisk.explanation}</p>
          {combinedRisk.weights_used && (
            <p className="combined-risk-weights">
              Weights used:{' '}
              {Object.entries(combinedRisk.weights_used)
                .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
                .join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default RiskBreakdown;