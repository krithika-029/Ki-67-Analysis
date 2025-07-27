#!/bin/bash

# Ki-67 Analyzer Pro Setup Script

echo "🚀 Setting up Ki-67 Analyzer Pro..."
echo "=================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    echo "   Visit: https://python.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ Python version: $(python3 --version)"
echo ""

# Setup Backend
echo "🐍 Setting up Backend..."
cd backend

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Backend setup complete!"
echo ""

# Setup Frontend
echo "⚛️  Setting up Frontend..."
cd ../frontend

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"
echo ""

# Create .env files
echo "📝 Creating environment files..."
cd ..

# Frontend .env
cat > frontend/.env << EOL
REACT_APP_API_URL=http://localhost:8000
EOL

# Backend .env
cat > backend/.env << EOL
FLASK_ENV=development
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=50MB
EOL

echo "✅ Environment files created!"
echo ""

echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start the Backend (Terminal 1):"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "2. Start the Frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "3. Open your browser to:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo ""
echo "📖 For more information, see README.md"
