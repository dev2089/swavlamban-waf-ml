import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, Globe, Clock, Target } from 'lucide-react';

function Analytics() {
  const [threatsByType, setThreatsByType] = useState([]);
  const [threatsBySeverity, setThreatsBySeverity] = useState([]);
  const [topAttackers, setTopAttackers] = useState([]);
  const [hourlyDistribution, setHourlyDistribution] = useState([]);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    const { data: threats } = await supabase.from('threats').select('*');

    if (threats) {
      const typeCount = {};
      const severityCount = {};
      const ipCount = {};
      const hourCount = {};

      threats.forEach((threat) => {
        typeCount[threat.threat_type] = (typeCount[threat.threat_type] || 0) + 1;
        severityCount[threat.severity] = (severityCount[threat.severity] || 0) + 1;
        ipCount[threat.source_ip] = (ipCount[threat.source_ip] || 0) + 1;

        const hour = new Date(threat.detected_at).getHours();
        hourCount[hour] = (hourCount[hour] || 0) + 1;
      });

      setThreatsByType(
        Object.entries(typeCount).map(([name, value]) => ({ name, value }))
      );

      setThreatsBySeverity(
        Object.entries(severityCount).map(([name, value]) => ({ name, value }))
      );

      setTopAttackers(
        Object.entries(ipCount)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([ip, count]) => ({ ip, count }))
      );

      setHourlyDistribution(
        Array.from({ length: 24 }, (_, i) => ({
          hour: `${i}:00`,
          threats: hourCount[i] || 0,
        }))
      );
    }
  };

  const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6'];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h2>
        <p className="text-gray-600">Comprehensive threat analysis and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2" />
            Threats by Type
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={threatsByType}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2" />
            Threat Severity Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={threatsBySeverity}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {threatsBySeverity.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Clock className="w-5 h-5 mr-2" />
            Hourly Threat Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={hourlyDistribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="threats" stroke="#8b5cf6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Globe className="w-5 h-5 mr-2" />
            Top Attackers
          </h3>
          <div className="space-y-3">
            {topAttackers.map((attacker, index) => (
              <div
                key={attacker.ip}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-danger-100 rounded-full flex items-center justify-center font-bold text-danger-700">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-900">{attacker.ip}</span>
                </div>
                <span className="px-3 py-1 bg-danger-100 text-danger-800 rounded-full text-sm font-medium">
                  {attacker.count} attacks
                </span>
              </div>
            ))}
            {topAttackers.length === 0 && (
              <p className="text-center text-gray-500 py-8">No data available</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Analytics;
