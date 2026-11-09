import { FileCode, Activity, Info, HelpCircle } from 'lucide-react';
import { motion } from 'framer-motion';

// Module 5: two independent K-Means clustering models predict which
// behavioral "family" archetype this APK's STATIC features (permissions/API
// calls) and FLOW features (network behavior) most resemble. These are
// cluster-based interpretations, not verified ground-truth family labels
// (no labeled family dataset exists for this project) - see
// family_predictor.py's module docstring for the full honesty notes this
// mirrors (same pattern as Module 3's FPR/detection tradeoff, Module 6's
// documented weighting choices).
const FamilyClassification = ({ classification, isAnalyzing }) => {
  const staticMatch = classification?.static;
  const flowMatch = classification?.flow;
  const hasFlowMatch = flowMatch && flowMatch.cluster_id != null;

  const ConfidenceBar = ({ confidence, color }) => (
    <div className="prob-bar">
      <div
        className="prob-fill"
        style={{
          width: `${Math.round((confidence || 0) * 100)}%`,
          background: `linear-gradient(90deg, ${color}, ${color}cc)`,
        }}
      />
    </div>
  );

  return (
    <div className="family-classification">
      <div className="section-header">
        <h3 className="section-title">Malware Family Classification</h3>
        <div className={`classification-badge ${!classification ? 'pending' : ''}`}>
          <Info size={14} />
          <span>Module 5: K-Means Clustering (static + behavioral)</span>
        </div>
      </div>

      <div className="families-grid">
        {/* --- Static profile card --- */}
        <motion.div
          className="family-card expanded"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          style={{ borderTopColor: '#4f46e5' }}
        >
          <div className="family-header">
            <div className="family-icon" style={{ background: '#4f46e520' }}>
              <FileCode size={30} color="#4f46e5" />
            </div>
            <div className="family-info">
              <div className="family-name-row">
                <h4>{staticMatch ? staticMatch.name : 'Static Profile'}</h4>
                <span className="family-risk" style={{ background: '#4f46e520', color: '#4f46e5' }}>
                  Permission / API-based
                </span>
              </div>
              {staticMatch && (
                <div className="family-probability">
                  <ConfidenceBar confidence={staticMatch.confidence} color="#4f46e5" />
                  <span className="prob-value">{Math.round(staticMatch.confidence * 100)}%</span>
                </div>
              )}
            </div>
          </div>

          <div className="family-details" style={{ height: 'auto', opacity: 1 }}>
            <div className="details-content">
              {staticMatch ? (
                <div className="detail-section">
                  <p>{staticMatch.description}</p>
                </div>
              ) : (
                <div className="detail-section">
                  <p>{isAnalyzing ? 'Classifying static profile…' : 'Awaiting analysis.'}</p>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* --- Flow / behavioral profile card --- */}
        <motion.div
          className="family-card expanded"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          style={{ borderTopColor: '#ec4899' }}
        >
          <div className="family-header">
            <div className="family-icon" style={{ background: '#ec489920' }}>
              <Activity size={30} color="#ec4899" />
            </div>
            <div className="family-info">
              <div className="family-name-row">
                <h4>{hasFlowMatch ? flowMatch.name : 'Behavioral Profile'}</h4>
                <span className="family-risk" style={{ background: '#ec489920', color: '#ec4899' }}>
                  Network Behavior
                </span>
              </div>
              {hasFlowMatch && (
                <div className="family-probability">
                  <ConfidenceBar confidence={flowMatch.confidence} color="#ec4899" />
                  <span className="prob-value">{Math.round(flowMatch.confidence * 100)}%</span>
                </div>
              )}
            </div>
          </div>

          <div className="family-details" style={{ height: 'auto', opacity: 1 }}>
            <div className="details-content">
              {hasFlowMatch ? (
                <>
                  <div className="detail-section">
                    <p>{flowMatch.description}</p>
                  </div>
                  <div className="detail-footer">
                    <div className="detail-meta">
                      <span>{flowMatch.flows_analyzed} network flow(s) analyzed</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="detail-section">
                  <p>
                    {isAnalyzing
                      ? 'Classifying behavioral profile…'
                      : flowMatch?.note || 'Awaiting analysis.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>

      <div className="combined-risk-footer">
        <p className="combined-risk-explanation">
          <HelpCircle size={14} style={{ verticalAlign: 'text-bottom', marginRight: '4px' }} />
          These are cluster-based interpretations from unsupervised K-Means models
          (Module 5), not verified ground-truth family labels — no labeled malware
          family dataset was available for this project. Confidence reflects relative
          closeness to the matched cluster's center, not a calibrated probability.
        </p>
      </div>

      {isAnalyzing && (
        <div className="analysis-loading">
          <div className="loading-spinner"></div>
          <p>Classifying malware family patterns...</p>
        </div>
      )}
    </div>
  );
};

export default FamilyClassification;
