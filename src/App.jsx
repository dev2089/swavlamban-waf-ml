import { useState, useEffect } from 'react';
import { supabase } from './lib/supabase';
import Dashboard from './components/Dashboard';
import ThreatMonitor from './components/ThreatMonitor';
import Analytics from './components/Analytics';
import RulesManager from './components/RulesManager';
import Header from './components/Header';
import Sidebar from './components/Sidebar';

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [stats, setStats] = useState({
    totalThreats: 0,
    blockedRequests: 0,
    activeRules: 0,
    detectionRate: 0,
  });

  useEffect(() => {
    fetchStats();

    // Subscribe to real-time threat updates
    const threatsSubscription = supabase
      .channel('threats')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'threats' }, () => {
        fetchStats();
      })
      .subscribe();

    return () => {
      threatsSubscription.unsubscribe();
    };
  }, []);

  const fetchStats = async () => {
    try {
      const [threatsResult, rulesResult, logsResult] = await Promise.all([
        supabase.from('threats').select('*', { count: 'exact', head: true }),
        supabase.from('security_rules').select('*', { count: 'exact', head: true }).eq('enabled', true),
        supabase.from('request_logs').select('blocked', { count: 'exact' }),
      ]);

      const blockedCount = logsResult.data?.filter(log => log.blocked).length || 0;
      const totalRequests = logsResult.count || 1;

      setStats({
        totalThreats: threatsResult.count || 0,
        blockedRequests: blockedCount,
        activeRules: rulesResult.count || 0,
        detectionRate: totalRequests > 0 ? ((blockedCount / totalRequests) * 100).toFixed(2) : 0,
      });
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard stats={stats} />;
      case 'threats':
        return <ThreatMonitor />;
      case 'analytics':
        return <Analytics />;
      case 'rules':
        return <RulesManager />;
      default:
        return <Dashboard stats={stats} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <div className="flex">
        <Sidebar activeView={activeView} setActiveView={setActiveView} />
        <main className="flex-1 p-8">
          {renderView()}
        </main>
      </div>
    </div>
  );
}

export default App;
