import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { Plus, Edit, Trash2, Power, PowerOff } from 'lucide-react';

function RulesManager() {
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);

  useEffect(() => {
    fetchRules();
  }, []);

  const fetchRules = async () => {
    setLoading(true);
    const { data } = await supabase
      .from('security_rules')
      .select('*')
      .order('created_at', { ascending: false });
    if (data) setRules(data);
    setLoading(false);
  };

  const toggleRule = async (ruleId, currentStatus) => {
    const { error } = await supabase
      .from('security_rules')
      .update({ enabled: !currentStatus })
      .eq('id', ruleId);

    if (!error) {
      setRules(rules.map((rule) =>
        rule.id === ruleId ? { ...rule, enabled: !currentStatus } : rule
      ));
    }
  };

  const deleteRule = async (ruleId) => {
    if (confirm('Are you sure you want to delete this rule?')) {
      const { error } = await supabase
        .from('security_rules')
        .delete()
        .eq('id', ruleId);

      if (!error) {
        setRules(rules.filter((rule) => rule.id !== ruleId));
      }
    }
  };

  const getThreatLevelColor = (level) => {
    const colors = {
      critical: 'badge-critical',
      high: 'badge-high',
      medium: 'badge-medium',
      low: 'badge-low',
    };
    return colors[level] || 'badge-medium';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Rules Manager</h2>
          <p className="text-gray-600">Configure security rules and policies</p>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>Add Rule</span>
        </button>
      </div>

      <div className="card">
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-300 border-t-primary-600"></div>
          </div>
        ) : (
          <div className="space-y-3">
            {rules.map((rule) => (
              <div
                key={rule.id}
                className="p-4 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-lg font-semibold text-gray-900">{rule.name}</h3>
                      <span className={`badge ${getThreatLevelColor(rule.threat_level)}`}>
                        {rule.threat_level}
                      </span>
                      <span className="text-sm text-gray-500">
                        Confidence: {(rule.confidence * 100).toFixed(0)}%
                      </span>
                    </div>

                    <p className="text-sm text-gray-600 mb-3">{rule.description}</p>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium text-gray-700">Type:</span>{' '}
                        <span className="text-gray-600">{rule.rule_type}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">Rule ID:</span>{' '}
                        <span className="text-gray-600">{rule.rule_id}</span>
                      </div>
                      <div className="col-span-2">
                        <span className="font-medium text-gray-700">Pattern:</span>
                        <pre className="mt-1 p-2 bg-gray-50 rounded text-xs overflow-x-auto">
                          {rule.pattern}
                        </pre>
                      </div>
                      <div className="col-span-2">
                        <span className="font-medium text-gray-700">Actions:</span>{' '}
                        {Array.isArray(rule.actions) && rule.actions.map((action) => (
                          <span key={action} className="ml-2 badge bg-blue-100 text-blue-800">
                            {action}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col space-y-2 ml-4">
                    <button
                      onClick={() => toggleRule(rule.id, rule.enabled)}
                      className={`p-2 rounded-lg transition-colors ${
                        rule.enabled
                          ? 'bg-green-100 hover:bg-green-200 text-green-700'
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                      }`}
                      title={rule.enabled ? 'Disable rule' : 'Enable rule'}
                    >
                      {rule.enabled ? <Power className="w-5 h-5" /> : <PowerOff className="w-5 h-5" />}
                    </button>
                    <button
                      onClick={() => deleteRule(rule.id)}
                      className="p-2 bg-danger-100 hover:bg-danger-200 text-danger-700 rounded-lg transition-colors"
                      title="Delete rule"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            ))}

            {rules.length === 0 && (
              <div className="text-center py-12">
                <p className="text-gray-500">No security rules configured</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default RulesManager;
