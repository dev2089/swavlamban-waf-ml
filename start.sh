#!/bin/bash

echo "======================================"
echo "  WAF ML - Starting Application"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file based on .env.example"
    echo ""
    echo "You need to add your Supabase credentials:"
    echo "  VITE_SUPABASE_URL=your_supabase_url"
    echo "  VITE_SUPABASE_ANON_KEY=your_supabase_anon_key"
    echo ""
    exit 1
fi

echo "Starting Backend Server..."
echo "Backend will run on http://localhost:8000"
echo ""

# Start backend in background
cd backend
python server.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

echo "Starting Frontend Development Server..."
echo "Frontend will run on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start frontend (this will run in foreground)
npm run dev

# When frontend stops, kill backend
kill $BACKEND_PID
echo ""
echo "Servers stopped."
