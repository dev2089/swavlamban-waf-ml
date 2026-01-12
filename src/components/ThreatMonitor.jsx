import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { format } from 'date-fns';
import { Shield, Search, Filter, AlertCircle } from 'lucide-react';

function ThreatMonitor() {
  const [threats, setThreats] = useState([]);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchThreats();

    const subscription = supabase
      .channel('threats-monitor')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'threats' }, () => {
        fetchThreats();
      })
      .subscribe();

    return () => subscription.unsubscribe();
  }, [filter]);

  const fetchThreats = async () => {
    setLoading(true);
    let query = supabase.from('threats').select('*').order('detected_at', { ascending: false });

    if (filter !== 'all') {
      query = query.eq('severity', filter);
    }

    const { data, error } = await query;
    if (error) {
      console.error('Error fetching threats:', error);
    } else {
      setThreats(data || []);
    }
    setLoading(false);
  };

  const filteredThreats = threats.filter((threat) =>
    threat.threat_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
    threat.source_ip.includes(searchTerm) ||
    threat.target_endpoint.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getSeverityColor = (severity) => {
    const colors = {
      critical: 'bg-danger-100 text-danger-800 border-danger-200',
      high: 'bg-orange-100 text-orange-800 border-orange-200',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      low: 'bg-green-100 text-green-800 border-green-200',
    };
    return colors[severity] || colors.medium;
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Threat Monitor</h2>
        <p className="text-gray-600">Real-time threat detection and analysis</p>
      </div>

      <div className="card">
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search threats..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div className="flex gap-2">
            {['all', 'critical', 'high', 'medium', 'low'].map((severity) => (
              <button
                key={severity}
                onClick={() => setFilter(severity)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  filter === severity
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {severity.charAt(0).toUpperCase() + severity.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-300 border-t-primary-600"></div>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredThreats.map((threat) => (
              <div
                key={threat.id}
                className={`p-4 rounded-lg border ${getSeverityColor(threat.severity)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <AlertCircle className="w-5 h-5" />
                      <span className="font-semibold">{threat.threat_type}</span>
                      <span className="text-sm">
                        Confidence: {(threat.confidence * 100).toFixed(0)}%
                      </span>
                      {threat.blocked && (
                        <span className="badge bg-red-600 text-white">Blocked</span>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Source IP:</span> {threat.source_ip}
                      </div>
                      <div>
                        <span className="font-medium">Target:</span> {threat.target_endpoint}
                      </div>
                      <div className="col-span-2">
                        <span className="font-medium">Payload:</span>
                        <pre className="mt-1 p-2 bg-white bg-opacity-50 rounded text-xs overflow-x-auto">
                          {threat.payload || 'N/A'}
                        </pre>
                      </div>
                    </div>
                  </div>

                  <div className="text-right text-sm">
                    <div className="text-gray-600">
                      {format(new Date(threat.detected_at), 'PPp')}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {filteredThreats.length === 0 && (
              <div className="text-center py-12">
                <Shield className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No threats found</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ThreatMonitor;
