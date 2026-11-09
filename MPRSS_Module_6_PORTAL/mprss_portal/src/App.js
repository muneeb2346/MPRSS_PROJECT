import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import './index.css';

function App() {
  const [activeItem, setActiveItem] = useState('dashboard');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleSidebarToggle = (collapsed) => {
    setIsSidebarCollapsed(collapsed);
  };

  return (
      <div className="App">
        <Sidebar 
          activeItem={activeItem} 
          setActiveItem={(item) => {
            setActiveItem(item);
            scrollToSection(item);
          }}
          onToggle={handleSidebarToggle}
        />
        <Dashboard isSidebarCollapsed={isSidebarCollapsed} />
      </div>
  );
}

export default App;