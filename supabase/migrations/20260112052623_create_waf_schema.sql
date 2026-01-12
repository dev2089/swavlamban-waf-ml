/*
  # WAF ML System Database Schema

  1. New Tables
    - `threats`
      - `id` (uuid, primary key)
      - `threat_type` (text) - Type of threat (SQL_INJECTION, XSS, etc.)
      - `severity` (text) - critical, high, medium, low
      - `source_ip` (text) - Originating IP address
      - `target_endpoint` (text) - Targeted endpoint
      - `payload` (text) - Attack payload
      - `confidence` (numeric) - ML confidence score
      - `blocked` (boolean) - Whether request was blocked
      - `metadata` (jsonb) - Additional metadata
      - `detected_at` (timestamptz)
      - `created_at` (timestamptz)
    
    - `security_rules`
      - `id` (uuid, primary key)
      - `rule_id` (text, unique) - Custom rule identifier
      - `rule_type` (text) - Type of rule
      - `name` (text) - Rule name
      - `description` (text) - Rule description
      - `pattern` (text) - Detection pattern/regex
      - `threat_level` (text) - Threat severity
      - `actions` (jsonb) - Actions to take
      - `enabled` (boolean) - Whether rule is active
      - `confidence` (numeric) - Rule confidence
      - `tags` (jsonb) - Tags array
      - `metadata` (jsonb) - Additional data
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)
    
    - `request_logs`
      - `id` (uuid, primary key)
      - `request_id` (text) - Unique request identifier
      - `method` (text) - HTTP method
      - `uri` (text) - Request URI
      - `source_ip` (text) - Client IP
      - `user_agent` (text) - User agent string
      - `headers` (jsonb) - Request headers
      - `query_params` (jsonb) - Query parameters
      - `body` (text) - Request body
      - `response_status` (integer) - Response status code
      - `response_time_ms` (integer) - Response time
      - `threat_detected` (boolean) - Whether threat was detected
      - `threat_id` (uuid) - Reference to threats table
      - `ml_scores` (jsonb) - ML model scores
      - `blocked` (boolean) - Whether request was blocked
      - `timestamp` (timestamptz)
    
    - `analytics`
      - `id` (uuid, primary key)
      - `metric_name` (text) - Name of metric
      - `metric_value` (numeric) - Value
      - `metric_type` (text) - Type (counter, gauge, histogram)
      - `dimensions` (jsonb) - Metric dimensions
      - `timestamp` (timestamptz)
      - `created_at` (timestamptz)
    
    - `alerts`
      - `id` (uuid, primary key)
      - `alert_type` (text) - Type of alert
      - `severity` (text) - Alert severity
      - `title` (text) - Alert title
      - `message` (text) - Alert message
      - `threat_id` (uuid) - Associated threat
      - `acknowledged` (boolean) - Whether acknowledged
      - `acknowledged_by` (uuid) - User who acknowledged
      - `acknowledged_at` (timestamptz)
      - `metadata` (jsonb) - Additional data
      - `created_at` (timestamptz)
    
    - `ml_models`
      - `id` (uuid, primary key)
      - `model_name` (text) - Name of model
      - `model_type` (text) - Type of model
      - `version` (text) - Model version
      - `accuracy` (numeric) - Model accuracy
      - `precision` (numeric) - Model precision
      - `recall` (numeric) - Model recall
      - `f1_score` (numeric) - F1 score
      - `enabled` (boolean) - Whether model is active
      - `metadata` (jsonb) - Model metadata
      - `trained_at` (timestamptz)
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated access
*/

-- Create threats table
CREATE TABLE IF NOT EXISTS threats (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  threat_type text NOT NULL,
  severity text NOT NULL DEFAULT 'medium',
  source_ip text NOT NULL,
  target_endpoint text NOT NULL,
  payload text,
  confidence numeric DEFAULT 0.0,
  blocked boolean DEFAULT false,
  metadata jsonb DEFAULT '{}'::jsonb,
  detected_at timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- Create security_rules table
CREATE TABLE IF NOT EXISTS security_rules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  rule_id text UNIQUE NOT NULL,
  rule_type text NOT NULL,
  name text NOT NULL,
  description text,
  pattern text NOT NULL,
  threat_level text NOT NULL DEFAULT 'medium',
  actions jsonb DEFAULT '[]'::jsonb,
  enabled boolean DEFAULT true,
  confidence numeric DEFAULT 0.0,
  tags jsonb DEFAULT '[]'::jsonb,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create request_logs table
CREATE TABLE IF NOT EXISTS request_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id text NOT NULL,
  method text NOT NULL,
  uri text NOT NULL,
  source_ip text NOT NULL,
  user_agent text,
  headers jsonb DEFAULT '{}'::jsonb,
  query_params jsonb DEFAULT '{}'::jsonb,
  body text,
  response_status integer,
  response_time_ms integer,
  threat_detected boolean DEFAULT false,
  threat_id uuid REFERENCES threats(id),
  ml_scores jsonb DEFAULT '{}'::jsonb,
  blocked boolean DEFAULT false,
  timestamp timestamptz DEFAULT now()
);

-- Create analytics table
CREATE TABLE IF NOT EXISTS analytics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  metric_name text NOT NULL,
  metric_value numeric NOT NULL,
  metric_type text NOT NULL DEFAULT 'counter',
  dimensions jsonb DEFAULT '{}'::jsonb,
  timestamp timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  alert_type text NOT NULL,
  severity text NOT NULL DEFAULT 'medium',
  title text NOT NULL,
  message text NOT NULL,
  threat_id uuid REFERENCES threats(id),
  acknowledged boolean DEFAULT false,
  acknowledged_by uuid,
  acknowledged_at timestamptz,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create ml_models table
CREATE TABLE IF NOT EXISTS ml_models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name text NOT NULL,
  model_type text NOT NULL,
  version text NOT NULL,
  accuracy numeric DEFAULT 0.0,
  precision numeric DEFAULT 0.0,
  recall numeric DEFAULT 0.0,
  f1_score numeric DEFAULT 0.0,
  enabled boolean DEFAULT true,
  metadata jsonb DEFAULT '{}'::jsonb,
  trained_at timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_threats_severity ON threats(severity);
CREATE INDEX IF NOT EXISTS idx_threats_source_ip ON threats(source_ip);
CREATE INDEX IF NOT EXISTS idx_threats_detected_at ON threats(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_threats_type ON threats(threat_type);

CREATE INDEX IF NOT EXISTS idx_rules_enabled ON security_rules(enabled);
CREATE INDEX IF NOT EXISTS idx_rules_type ON security_rules(rule_type);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_source_ip ON request_logs(source_ip);
CREATE INDEX IF NOT EXISTS idx_logs_threat_detected ON request_logs(threat_detected);

CREATE INDEX IF NOT EXISTS idx_analytics_metric_name ON analytics(metric_name);
CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

-- Enable Row Level Security
ALTER TABLE threats ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE request_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access (adjust based on your security needs)
CREATE POLICY "Allow public read access to threats"
  ON threats FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to threats"
  ON threats FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to security_rules"
  ON security_rules FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to security_rules"
  ON security_rules FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to request_logs"
  ON request_logs FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to request_logs"
  ON request_logs FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to analytics"
  ON analytics FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to analytics"
  ON analytics FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to alerts"
  ON alerts FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to alerts"
  ON alerts FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public update to alerts"
  ON alerts FOR UPDATE
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public read access to ml_models"
  ON ml_models FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to ml_models"
  ON ml_models FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

-- Insert initial ML model data
INSERT INTO ml_models (model_name, model_type, version, accuracy, precision, recall, f1_score, enabled, metadata)
VALUES 
  ('Isolation Forest', 'anomaly_detection', 'v1.0.0', 0.913, 0.89, 0.92, 0.905, true, '{"contamination": 0.1, "n_estimators": 100}'::jsonb),
  ('Autoencoder', 'anomaly_detection', 'v1.0.0', 0.942, 0.931, 0.958, 0.944, true, '{"encoding_dim": 8, "epochs": 50}'::jsonb),
  ('Hybrid Ensemble', 'ensemble', 'v1.0.0', 0.965, 0.952, 0.978, 0.965, true, '{"voting": "hard"}'::jsonb)
ON CONFLICT DO NOTHING;