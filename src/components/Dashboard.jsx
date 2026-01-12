import { useState, useEffect } from 'react';
import { Shield, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { format } from 'date-fns';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function Dashboard({ stats }) {
  const [recentThreats, setRecentThreats] = useState([]);
  const [threatTrend, setThreatTrend] = useState([]);

  useEffect(() => {
    fetchRecentThreats();
    fetchThreatTrend();

    const subscription = supabase
      .channel('threats-dashboard')
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'threats' }, (payload) => {
        setRecentThreats((prev) => [payload.new, ...prev.slice(0, 4)]);
      })
      .subscribe();

    return () => subscription.unsubscribe();
  }, []);

  const fetchRecentThreats = async () => {
    const { data } = await supabase
      .from('threats')
      .select('*')
      .order('detected_at', { ascending: false })
      .limit(5);
    if (data) setRecentThreats(data);
  };

  const fetchThreatTrend = async () => {
    const { data } = await supabase
      .from('analytics')
      .select('*')
      .eq('metric_name', 'threat_count')
      .order('timestamp', { ascending: true })
      .limit(24);

    if (data) {
      const formattedData = data.map((item) => ({
        time: format(new Date(item.timestamp), 'HH:mm'),
        threats: item.metric_value,
      }));
      setThreatTrend(formattedData);
    }
  };

  const statCards = [
    {
      title: 'Total Threats',
      value: stats.totalThreats,
      icon: AlertTriangle,
      color: 'danger',
      change: '+12%',
    },
    {
      title: 'Blocked Requests',
      value: stats.blockedRequests,
      icon: Shield,
      color: 'primary',
      change: '+8%',
    },
    {
      title: 'Active Rules',
      value: stats.activeRules,
      icon: CheckCircle,
      color: 'green',
      change: '+3',
    },
    {
      title: 'Detection Rate',
      value: `${stats.detectionRate}%`,
      icon: Activity,
      color: 'blue',
      change: '+2.5%',
    },
  ];

  const getSeverityColor = (severity) => {
    const colors = {
      critical: 'badge-critical',
      high: 'badge-high',
      medium: 'badge-medium',
      low: 'badge-low',
    };
    return colors[severity] || 'badge-medium';
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Security Dashboard</h2>
        <p className="text-gray-600">Real-time threat monitoring and analytics</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat) => (
          <div key={stat.title} className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stat.value}</p>
                <p className="text-sm text-green-600 mt-1">{stat.change} from last hour</p>
              </div>
              <div className={`p-3 bg-${stat.color}-100 rounded-lg`}>
                <stat.icon className={`w-8 h-8 text-${stat.color}-600`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Threat Trend (24h)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={threatTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="threats" stroke="#0ea5e9" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Threats</h3>
          <div className="space-y-3">
            {recentThreats.map((threat) => (
              <div
                key={threat.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200"
              >
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className={`badge ${getSeverityColor(threat.severity)}`}>
                      {threat.severity}
                    </span>
                    <span className="text-sm font-medium text-gray-900">{threat.threat_type}</span>
                  </div>
                  <p className="text-xs text-gray-600 mt-1">
                    {threat.source_ip} → {threat.target_endpoint}
                  </p>
                </div>
                <div className="text-xs text-gray-500">
                  {format(new Date(threat.detected_at), 'HH:mm:ss')}
                </div>
              </div>
            ))}
            {recentThreats.length === 0 && (
              <p className="text-center text-gray-500 py-4">No threats detected</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
