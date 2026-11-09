import { useEffect, useRef } from 'react';
import { Terminal, Activity } from 'lucide-react';

const LiveLog = ({ logs, isActive }) => {
  const logEndRef = useRef(null);
  const logContainerRef = useRef(null);

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [logs]);

  const getLogIcon = (log) => {
    if (log.includes('ERROR') || log.includes('ALERT')) return '🔴';
    if (log.includes('SUCCESS') || log.includes('Complete') || log.includes('✅')) return '✅';
    if (log.includes('UPLOADED') || log.includes('extracting')) return '📦';
    if (log.includes('DETECTED')) return '⚠️';
    if (log.includes('[STATIC]')) return '📊';
    if (log.includes('[BEHAVIORAL]')) return '🎯';
    if (log.includes('[ANOMALY]')) return '🚨';
    if (log.includes('[CLASSIFIER]')) return '🤖';
    if (log.includes('[SHAP]')) return '💡';
    return '>';
  };

  const getLogColor = (message) => {
    if (message.includes('ERROR') || message.includes('ALERT')) return '#ff6b6b';
    if (message.includes('SUCCESS') || message.includes('Complete') || message.includes('✅')) return '#51cf66';
    if (message.includes('DETECTED')) return '#ffd43b';
    if (message.includes('[STATIC]')) return '#74c0fc';
    if (message.includes('[BEHAVIORAL]')) return '#ff8787';
    if (message.includes('[ANOMALY]')) return '#ff922b';
    if (message.includes('[CLASSIFIER]')) return '#b197fc';
    if (message.includes('[SHAP]')) return '#94d82d';
    return '#00ff88';
  };

  // Generate sample logs for testing scrollbar (remove in production)
  const sampleLogs = logs.length === 0 && !isActive ? [
    { time: '14:23:45', message: 'Ready for analysis. Upload an APK file to begin.' },
    { time: '14:23:46', message: 'System idle. Waiting for input...' },
  ] : logs;

  const displayLogs = logs.length === 0 && !isActive ? sampleLogs : logs;

  return (
    <div className="live-log">
      <div className="log-header">
        <div className="log-title">
          <Terminal size={36} strokeWidth={1.5} />
          <h3>Live Analysis Log</h3>
        </div>
        <div className={`log-status ${isActive ? 'active' : ''}`}>
          <Activity size={20} />
          <span>{isActive ? '● ACTIVE' : '○ IDLE'}</span>
        </div>
      </div>
      
      <div className="log-content-wrapper">
        <div className="log-content" ref={logContainerRef}>
          <div className="log-entries-wrapper">
            {displayLogs.length === 0 ? (
              <div className="log-empty">
                <p>📭 No analysis running</p>
                <p className="log-hint">Upload an APK and click ANALYZE to start</p>
              </div>
            ) : (
              <>
                {displayLogs.map((log, index) => (
                  <div 
                    key={index} 
                    className="log-entry" 
                    style={{ 
                      animationDelay: `${index * 0.02}s`,
                    }}
                  >
                    <span className="log-time">{log.time || new Date().toLocaleTimeString()}</span>
                    <span className="log-icon">{getLogIcon(log.message)}</span>
                    <span className="log-message" style={{ color: getLogColor(log.message) }}>
                      {log.message}
                    </span>
                  </div>
                ))}
                <div ref={logEndRef} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveLog;