import { useState, useCallback } from 'react';
import { Upload, FileArchive, X, CheckCircle } from 'lucide-react';

const UploadArea = ({ onFileUpload, isAnalyzing }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const validateFile = (file) => {
    if (!file.name.endsWith('.apk')) {
      setError('Please upload a valid .apk file');
      return false;
    }
    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      setError('File size must be less than 100MB');
      return false;
    }
    setError('');
    return true;
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (validateFile(file)) {
      setSelectedFile(file);
    }
  }, []);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (validateFile(file)) {
      setSelectedFile(file);
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    } else {
      setError('Please select an APK file first');
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setError('');
  };

  return (
    <div className="upload-area">
      <div className="upload-container">
        <div
          className={`drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!selectedFile ? (
            <>
              <div className="upload-icon">
                <Upload size={48} strokeWidth={1.5} />
              </div>
              <h3 style={{fontSize: '28px'}}>Drag & Drop</h3>
              <p className="file-type" style={{fontSize: '33px', fontWeight: 'bold'}}>.APK File Here</p>
              <p className="or-text" style={{ fontSize: '30px' }}>or</p>
                <label className="browse-btn" style={{ fontSize: '35px', padding: '14px 35px' }}>
                click to browse
                <input
                  type="file"
                  accept=".apk"
                  onChange={handleChange}
                  style={{ display: 'none' }}
                />
              </label>
              <p className="file-limit">Maximum file size: 100MB</p>
            </>
          ) : (
            <div className="file-info">
              <FileArchive size={50} />
              <div className="file-details">
                <p className="file-name" style={{ fontSize: '16px', fontWeight: 'bold' }}>{selectedFile.name}</p>
                <p className="file-size" style={{ fontSize: '14px' }}>
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
              <button className="remove-file" onClick={removeFile}>
                <X size={30} />
              </button>
            </div>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}

        <button
          className={`analyze-btn ${isAnalyzing ? 'analyzing' : ''}`}
          style={{ fontSize: '35px', padding: '16px' }}
          onClick={handleAnalyze}
          disabled={!selectedFile || isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <div className="spinner"></div>
              Analyzing...
            </>
          ) : (
            <>
              <CheckCircle size={20} />
              ANALYZE
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default UploadArea;