import React, { useState } from 'react';
import UploadArea from './UploadArea';
import RiskScore from './RiskScore';
import RiskBreakdown from './RiskBreakdown';
import LiveLog from './LiveLog';
import FamilyClassification from './FamilyClassification';
import XaiInsights from './XaiInsights';

// Module 7 orchestrator: ties static analysis (Module 2), live network
// capture + anomaly scoring (Module 3), and behavioral hooking (Frida)
// together behind one endpoint. See module7_orchestrator.py.
const API_BASE = 'http://localhost:5002';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const Dashboard = ({ isSidebarCollapsed }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasAnalysisStarted, setHasAnalysisStarted] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  const [riskData, setRiskData] = useState({
    score: 0,
    isMalicious: false,
    anomalyAlert: false,
    confidence: null
  });

  // NOTE: 'static' (Module 2) and 'anomaly' (Module 3) are real, live model
  // scores from the orchestrator. 'behavioral' stays null — Module 1's
  // RNN/LSTM isn't trained yet, so there's no score to show, even though real
  // behavioral EVENTS are captured (see behavioralInfo below and the live
  // log). 'mlConfidence' now holds Module 6's combined_score (see
  // risk_scorer.py / combinedRisk state below) rather than being hardcoded
  // to null. RiskBreakdown.jsx renders null as "Pending", not 0%, so the
  // dashboard never implies a score that doesn't exist yet.
  const [breakdownScores, setBreakdownScores] = useState({
    static: 0,
    behavioral: null,
    anomaly: null,
    mlConfidence: null
  });

  // Real behavioral event data from Module 7's Frida hooks — not a score,
  // just a factual count/summary of what was actually observed at runtime.
  const [behavioralInfo, setBehavioralInfo] = useState(null);

  // Module 6's combined risk score — comes back as `data.combined_risk` from
  // the orchestrator (see risk_scorer.py: compute_combined_risk()). Holds
  // combined_score, risk_level, components_used, weights_used, explanation.
  const [combinedRisk, setCombinedRisk] = useState(null);

  const [familyPredictions, setFamilyPredictions] = useState(null);
  const [xaiFeatures, setXaiFeatures] = useState(null);
  const [logs, setLogs] = useState([]);

  const addLog = (message) => {
    const time = new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
    setLogs(prev => [...prev, { time, message }]);
  };

  const analyzeAPK = async (file) => {
    setIsAnalyzing(true);
    setHasAnalysisStarted(true);
    setAnalysisError(null);
    setLogs([]);
    setFamilyPredictions(null);
    setXaiFeatures(null);
    setBehavioralInfo(null);
    setCombinedRisk(null);

    addLog(`APK uploaded: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)}MB)`);
    await sleep(300);

    addLog('Sending APK to sandbox orchestrator (Module 7): install → ' +
           'behavioral hooks → network capture → static + anomaly scoring...');
    addLog('This runs the app in the emulator for ~2 minutes — please wait.');

    const formData = new FormData();
    formData.append('apk', file);

    try {
      const response = await fetch(`${API_BASE}/api/analyze/full`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || `Server error (HTTP ${response.status})`);
      }

      const data = await response.json();
      const staticResult = data.static_analysis || {};
      const anomalyResult = data.anomaly_analysis || {};
      const behavioralSummary = data.behavioral_summary || {};
      const combined = data.combined_risk || {};
      const familyResult = data.family_classification || {};

      if (staticResult.error) {
        throw new Error(`Static analysis failed: ${staticResult.error}`);
      }

      // --- Static (Module 2) ---
      addLog(`[STATIC] Parsed ${staticResult.app_info?.package_name} (${staticResult.app_info?.app_name})`);
      await sleep(300);
      addLog(`[STATIC] ${staticResult.app_info?.permissions_declared} permissions declared, ` +
             `${staticResult.app_info?.total_active_features} risk-relevant features matched`);
      await sleep(300);
      addLog(`[STATIC] Random Forest static risk score: ${staticResult.static_score}/100`);
      await sleep(300);

      // --- Anomaly (Module 3) ---
      if (anomalyResult.anomaly_score != null) {
        addLog(`[ANOMALY] ${anomalyResult.flows_analyzed} network flow(s) captured and scored`);
        await sleep(300);
        addLog(`[ANOMALY] One-Class SVM anomaly score: ${anomalyResult.anomaly_score}/100 ` +
               `(${anomalyResult.flows_flagged} of ${anomalyResult.flows_analyzed} flows flagged)`);
      } else {
        addLog(`[ANOMALY] ${anomalyResult.note || 'No network flows captured for this run.'}`);
      }
      await sleep(300);

      // --- Combined risk (Module 6) ---
      if (combined.combined_score != null) {
        addLog(`[RISK] Module 6 combined score: ${combined.combined_score}/100 (${combined.risk_level}) — ` +
               `${combined.explanation}`);
      } else {
        addLog('[RISK] Module 6 could not compute a combined score (no components available).');
      }
      await sleep(300);

      // --- Family classification (Module 5) ---
      if (familyResult.static?.name) {
        addLog(`[FAMILY] Static profile match: ${familyResult.static.name} ` +
               `(${(familyResult.static.confidence * 100).toFixed(0)}% confidence)`);
      }
      if (familyResult.flow?.name) {
        addLog(`[FAMILY] Behavioral profile match: ${familyResult.flow.name} ` +
               `(${familyResult.flow.flows_analyzed} flows analyzed)`);
      } else if (familyResult.flow?.note) {
        addLog(`[FAMILY] ${familyResult.flow.note}`);
      }
      await sleep(300);
      setFamilyPredictions(familyResult);

      // --- Saved network capture (kept on disk for manual inspection in Wireshark) ---
      if (data.pcap_saved_path) {
        addLog(`[CAPTURE] Network traffic saved to: ${data.pcap_saved_path}`);
        await sleep(300);
      }

      // --- Behavioral (Module 7's Frida hooks — real events, no trained model yet) ---
      addLog(`[BEHAVIORAL] ${behavioralSummary.total_events || 0} real API event(s) observed ` +
             `during execution (permissions, telephony, network, file I/O, etc.)`);
      if (behavioralSummary.total_events > 0) {
        const catSummary = Object.entries(behavioralSummary.events_by_category || {})
          .map(([cat, count]) => `${cat}: ${count}`)
          .join(', ');
        addLog(`[BEHAVIORAL] By category — ${catSummary}`);
      }
      await sleep(300);

      setBehavioralInfo({
        totalEvents: behavioralSummary.total_events || 0,
        eventsByCategory: behavioralSummary.events_by_category || {},
        events: data.behavioral_events || []
      });

      setCombinedRisk(combined.combined_score != null ? combined : null);

      setBreakdownScores({
        static: staticResult.static_score,
        behavioral: null, // Module 1 model not trained yet — see behavioralInfo for real event data
        anomaly: anomalyResult.anomaly_score ?? null,
        // Module 6's combined score (static+anomaly+behavioral, weighted).
        // This card now reflects the actual output of risk_scorer.py rather
        // than a hardcoded null.
        mlConfidence: combined.combined_score ?? null
      });

      if (staticResult.top_features && staticResult.top_features.length > 0) {
        const features = staticResult.top_features.map(f => ({
          name: f.feature,
          importance: f.importance,
          impact: 'positive',
          description: `${f.category} — contributes to the Random Forest static risk score`
        }));
        setXaiFeatures(features);
        addLog('[XAI] Top contributing static features identified');
        await sleep(300);
      }

      // Main risk gauge now shows Module 6's combined score (static +
      // anomaly, weighted and renormalized — see risk_scorer.py). Falls
      // back to the raw static score only if Module 6 couldn't compute
      // anything (e.g. both static and anomaly came back null).
      const hasCombined = combined.combined_score != null;
      setRiskData({
        score: hasCombined ? combined.combined_score : staticResult.static_score,
        isMalicious: hasCombined
          ? (combined.risk_level === 'High' || combined.risk_level === 'Critical')
          : staticResult.prediction === 'malicious',
        anomalyAlert: anomalyResult.is_anomaly === true,
        confidence: staticResult.confidence
      });

      addLog(`Static classification: ${(staticResult.prediction || 'unknown').toUpperCase()}`);
      await sleep(300);
      addLog(hasCombined
        ? `✅ Full sandbox analysis complete. Risk score reflects Module 6's ` +
          `combined score across ${combined.components_used.join(' + ')}.`
        : '✅ Full sandbox analysis complete. Module 6 had no components to ' +
          'combine — falling back to the raw static score.');

    } catch (err) {
      console.error('Analysis failed:', err);
      const message = err.message.includes('Failed to fetch')
        ? 'Could not reach the analysis server. Is the Module 7 orchestrator running on localhost:5002?'
        : err.message;
      setAnalysisError(message);
      addLog(`❌ ERROR: ${message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = (file) => {
    analyzeAPK(file);
  };

  const handleReset = () => {
    setHasAnalysisStarted(false);
    setAnalysisError(null);
    setLogs([]);
    setRiskData({ score: 0, isMalicious: false, anomalyAlert: false, confidence: null });
    setBreakdownScores({ static: 0, behavioral: null, anomaly: null, mlConfidence: null });
    setFamilyPredictions(null);
    setXaiFeatures(null);
    setBehavioralInfo(null);
    setCombinedRisk(null);
  };

  return (
    <div className={`dashboard ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <div className="dashboard-header">
        <h1>MPRSS Analysis Portal</h1>
        <p>Malware Prediction & Risk Scoring System</p>
      </div>

      <div className="upload-log-row">
        <div className="upload-section">
          <UploadArea onFileUpload={handleFileUpload} isAnalyzing={isAnalyzing} />
        </div>
        <div className="log-section">
          <LiveLog logs={logs} isActive={isAnalyzing} isAnalyzing={isAnalyzing} />
        </div>
      </div>

      {analysisError && (
        <div className="analysis-error-banner">
          <strong>Analysis failed:</strong> {analysisError}
        </div>
      )}

      {hasAnalysisStarted && (
        <>
          <div className="risk-assessment-row">
            <div className="risk-gauge-section">
              <RiskScore
                score={riskData.score}
                isMalicious={riskData.isMalicious}
                anomalyAlert={riskData.anomalyAlert}
                confidence={riskData.confidence}
              />
            </div>
            <div className="risk-breakdown-section">
              <RiskBreakdown scores={breakdownScores} combinedRisk={combinedRisk} />
            </div>
          </div>

          <div className="full-width-section">
            <FamilyClassification
              classification={familyPredictions}
              isAnalyzing={isAnalyzing}
            />
          </div>

          <div className="full-width-section">
            <XaiInsights
              features={xaiFeatures}
              isAnalyzing={isAnalyzing}
              confidence={riskData.confidence}
            />
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
