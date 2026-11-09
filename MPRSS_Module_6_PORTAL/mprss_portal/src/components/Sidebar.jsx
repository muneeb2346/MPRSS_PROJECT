import { useState } from 'react';
import { 
  LayoutDashboard, 
  Upload, 
  FileText, 
  Lightbulb, 
  GitBranch, 
  Settings,
  Shield
} from 'lucide-react';

const Sidebar = ({ activeItem, setActiveItem, onToggle }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'upload', label: 'Upload APK', icon: Upload },
    { id: 'reports', label: 'Reports', icon: FileText },
    { id: 'xai', label: 'XAI Insights', icon: Lightbulb },
    { id: 'clustering', label: 'Clustering', icon: GitBranch },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  const handleToggle = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    if (onToggle) {
      onToggle(newState);
    }
  };

  return (
    <div className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="logo-section">
        <div className="logo-icon">
          <Shield size={48} color="#4f46e5" />
        </div>
        {!isCollapsed && (
          <div className="logo-text">
            <h2>MPRSS</h2>
            <p>Analysis Portal</p>
          </div>
        )}
        <button className="collapse-btn" onClick={handleToggle}>
          {isCollapsed ? '→' : '←'}
        </button>
      </div>

      <nav className="nav-menu">
        {menuItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`nav-item ${activeItem === item.id ? 'active' : ''}`}
              onClick={() => setActiveItem(item.id)}
            >
              <Icon size={34} />
              {!isCollapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {!isCollapsed && (
        <div className="sidebar-footer">
          <p>Version 2.4.1</p>
          <p>Model: XGBoost v3.2</p>
        </div>
      )}
    </div>
  );
};

export default Sidebar;