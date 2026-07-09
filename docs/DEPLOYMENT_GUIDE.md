# TrackMyPDB Agentic Layer - Deployment Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Local Deployment](#local-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Docker Deployment](#docker-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python:** 3.8+ (3.10+ recommended)
- **OS:** Linux, macOS, or Windows
- **RAM:** 2GB minimum (4GB+ recommended)
- **Disk Space:** 2GB for dependencies
- **Network:** Internet connection required

### API Keys

1. **Anthropic API Key**
   - Visit: https://console.anthropic.com/
   - Create account or login
   - Generate API key
   - Copy and save securely

### Optional Tools

- **Docker** - For containerized deployment
- **Git** - For cloning repository
- **Conda** - For advanced dependency management

---

## Installation

### Step 1: Prepare Environment

```bash
# Create a working directory
mkdir trackmypdb-agent
cd trackmypdb-agent

# Clone or download TrackMyPDB
git clone https://github.com/Standard-Seed-Corporation/TrackMyPDB-Open-Source.git

# Or extract the ZIP file
unzip TrackMyPDB-Open-Source-sakeer.zip
```

### Step 2: Copy Agent Files

Copy these files to your working directory:
- `mcp_trackmypdb_server.py`
- `trackmypdb_agent.py`
- `chat_interface.py`
- `requirements_agent.txt`

Your directory should look like:
```
trackmypdb-agent/
├── TrackMyPDB-Open-Source-sakeer/
│   ├── backend/
│   ├── streamlit_app.py
│   └── requirements.txt
├── mcp_trackmypdb_server.py
├── trackmypdb_agent.py
├── chat_interface.py
└── requirements_agent.txt
```

### Step 3: Create Virtual Environment

```bash
# Using venv (built-in)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n trackmypdb python=3.10
conda activate trackmypdb
```

### Step 4: Install Dependencies

```bash
# Install TrackMyPDB requirements
pip install -r TrackMyPDB-Open-Source-sakeer/requirements.txt

# Install agent requirements
pip install -r requirements_agent.txt

# Verify installation
pip list | grep -E "(anthropic|streamlit|rdkit)"
```

### Step 5: Verify Installation

```bash
python test_agent.py
```

Expected output:
```
✓ File Structure PASSED
✓ MCP Server PASSED
✓ Agent Initialization (may require API key)
✓ Streamlit Interface PASSED
```

---

## Configuration

### Method 1: Environment Variable (Recommended)

```bash
# Linux/macOS
export ANTHROPIC_API_KEY='sk-your-key-here'

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY='sk-your-key-here'

# Verify
echo $ANTHROPIC_API_KEY
```

### Method 2: .env File

Create `.env` file in your working directory:
```
ANTHROPIC_API_KEY=sk-your-key-here
```

Then load it:
```bash
pip install python-dotenv

# In Python
from dotenv import load_dotenv
load_dotenv()
```

### Method 3: In Application

Set via the Streamlit interface sidebar when launching the app.

### Optional Configurations

```bash
# Specify Claude model
export CLAUDE_MODEL='claude-opus-4-6'

# Set MCP server port (for standalone server)
export MCP_SERVER_PORT=8000

# Enable debug logging
export DEBUG=1
```

---

## Verification

### Test 1: Import Modules

```python
python -c "from trackmypdb_agent import TrackMyPDBAgent; print('✓ Agent imported')"
python -c "from mcp_trackmypdb_server import MCPServer; print('✓ Server imported')"
```

### Test 2: Initialize Agent

```bash
python << 'EOF'
import os
os.environ['ANTHROPIC_API_KEY'] = 'sk-...'  # Set your key

from trackmypdb_agent import TrackMyPDBAgent
agent = TrackMyPDBAgent()
print("✓ Agent initialized successfully")
print(f"✓ Model: {agent.model}")
print(f"✓ Tools: {len(agent.tools)}")
EOF
```

### Test 3: Test MCP Server

```bash
python << 'EOF'
from mcp_trackmypdb_server import MCPServer
server = MCPServer()
tools = server.define_tools()
print(f"✓ MCP Server ready")
print(f"✓ Available tools: {len(tools)}")
for tool in tools:
    print(f"  - {tool['name']}")
EOF
```

### Test 4: Run Full Test Suite

```bash
python test_agent.py
```

---

## Local Deployment

### Launch Chat Interface

```bash
# Start Streamlit app
streamlit run chat_interface.py

# Alternative with port specification
streamlit run chat_interface.py --server.port 8501
```

Access the interface at: **http://localhost:8501**

### Interactive CLI Mode

For command-line interaction:

```bash
python trackmypdb_agent.py
```

Then type your questions:
```
You: Analyze protein Q9UNQ0
You: Extract heteroatoms
You: quit
```

### As Python Library

```python
from trackmypdb_agent import TrackMyPDBAgent

agent = TrackMyPDBAgent()

# Single analysis
response = agent.chat("What heteroatoms are in Q9UNQ0?")
print(response)

# Save results
agent.save_conversation("my_research.json")
```

---

## Cloud Deployment

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "Deploy an app"
   - Select your repository
   - Set main file: `chat_interface.py`

3. **Set Secrets**
   - In Streamlit Cloud settings, add:
   ```toml
   [secrets]
   ANTHROPIC_API_KEY = "sk-..."
   ```

4. **Access Your App**
   ```
   https://your-username-trackmypdb-agent.streamlit.app/
   ```

### Heroku Deployment

1. **Create Procfile**
   ```
   web: streamlit run chat_interface.py --server.port $PORT
   ```

2. **Create requirements.txt**
   ```
   # Combine both requirements files
   -r requirements_agent.txt
   -r TrackMyPDB-Open-Source-sakeer/requirements.txt
   ```

3. **Deploy**
   ```bash
   heroku create your-app-name
   heroku config:set ANTHROPIC_API_KEY=sk-...
   git push heroku main
   ```

### AWS Deployment

1. **Using EC2**
   ```bash
   # SSH into instance
   ssh -i key.pem ubuntu@instance-ip
   
   # Install Python
   sudo apt-get update
   sudo apt-get install python3.10 python3-pip
   
   # Clone repo and install
   git clone <your-repo>
   cd trackmypdb-agent
   pip install -r requirements_agent.txt
   
   # Run
   streamlit run chat_interface.py --server.address 0.0.0.0
   ```

2. **Using Elastic Beanstalk**
   ```bash
   eb init -p python-3.10 trackmypdb-agent
   eb create trackmypdb-agent
   eb setenv ANTHROPIC_API_KEY=sk-...
   eb deploy
   ```

3. **Using Lambda + API Gateway**
   - Use AWS SAM framework
   - Create serverless deployment

---

## Docker Deployment

### Option 1: Using Existing Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements_agent.txt .
COPY TrackMyPDB-Open-Source-sakeer/ ./TrackMyPDB-Open-Source-sakeer/
COPY *.py ./

# Install dependencies
RUN pip install -r requirements_agent.txt
RUN pip install -r TrackMyPDB-Open-Source-sakeer/requirements.txt

# Expose port
EXPOSE 8501

# Set environment
ENV PYTHONUNBUFFERED=1

# Run Streamlit
CMD ["streamlit", "run", "chat_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option 2: Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  trackmypdb-agent:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t trackmypdb-agent .

# Run container
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY='sk-...' \
  trackmypdb-agent

# Or with Docker Compose
docker-compose up
```

### Push to Docker Hub

```bash
docker tag trackmypdb-agent username/trackmypdb-agent
docker push username/trackmypdb-agent
```

---

## Production Checklist

- [ ] API key securely stored (environment variable, not hardcoded)
- [ ] Dependencies locked to specific versions
- [ ] Error logging configured
- [ ] Rate limiting implemented
- [ ] Session persistence enabled
- [ ] HTTPS enabled (for cloud deployment)
- [ ] Monitoring and alerting setup
- [ ] Database backup configured
- [ ] Documentation updated
- [ ] Tests passing

---

## Monitoring & Maintenance

### Monitor Application

```bash
# Streamlit logs
tail -f /path/to/streamlit/logs

# System resources
watch nvidia-smi  # If using GPU
htop  # CPU and memory
```

### Update Dependencies

```bash
# Check for updates
pip list --outdated

# Update packages
pip install --upgrade -r requirements_agent.txt

# Verify no breaking changes
python test_agent.py
```

### Backup Sessions

```bash
# Backup saved conversations
tar -czf backup_$(date +%Y%m%d).tar.gz *.json

# Store in cloud
aws s3 cp backup_*.tar.gz s3://my-bucket/
```

---

## Troubleshooting

### Issue: API Key Error
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# Set if missing
export ANTHROPIC_API_KEY='sk-...'

# Test connection
python -c "import anthropic; print('OK')"
```

### Issue: Module Not Found
```bash
# Verify installation
pip list | grep -E "(anthropic|streamlit|rdkit)"

# Reinstall if needed
pip install --force-reinstall -r requirements_agent.txt

# Check Python version
python --version  # Should be 3.8+
```

### Issue: Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run chat_interface.py --server.port 8502
```

### Issue: Out of Memory
```bash
# Reduce model size (use Haiku instead of Opus)
# In trackmypdb_agent.py, change:
# model="claude-haiku-4-5"

# Clear cache
rm -rf .streamlit/cache
```

### Issue: Slow Performance
```bash
# Check network
ping google.com

# Monitor API calls
# Add logging to agent

# Reduce batch size
# Process fewer proteins at once
```

---

## Advanced Configuration

### Custom System Prompt

Edit `trackmypdb_agent.py`:
```python
def _create_system_prompt(self) -> str:
    return """Your custom instructions..."""
```

### Integration with Other Tools

```python
# Extend MCP server with custom tools
class CustomMCPServer(MCPServer):
    def define_tools(self):
        tools = super().define_tools()
        # Add custom tools
        return tools
```

### Rate Limiting

```python
from functools import wraps
import time

def rate_limit(calls_per_minute=60):
    def decorator(func):
        last_called = [0.0]
        min_interval = 60.0 / calls_per_minute
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

---

## Performance Optimization

### Caching Results

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_heteroatom_extraction(uniprot_id):
    # Cached results
    pass
```

### Async Operations

```python
import asyncio

async def batch_analysis(proteins):
    tasks = [extract_heteroatoms_async(p) for p in proteins]
    return await asyncio.gather(*tasks)
```

### Database Storage

```python
import sqlite3

def save_results(results):
    conn = sqlite3.connect('results.db')
    # Store results in database
```

---

## Security Hardening

```python
# Input validation
def validate_smiles(smiles):
    if len(smiles) > 500:  # Max length
        raise ValueError("SMILES too long")
    if not all(c in allowed_chars for c in smiles):
        raise ValueError("Invalid SMILES characters")

# Rate limiting per user
from collections import defaultdict
user_requests = defaultdict(list)

def check_rate_limit(user_id, limit=10, window=60):
    now = time.time()
    user_requests[user_id] = [t for t in user_requests[user_id] 
                              if now - t < window]
    if len(user_requests[user_id]) >= limit:
        raise ValueError("Rate limit exceeded")
    user_requests[user_id].append(now)
```

---

## Support & Documentation

For issues or questions:
1. Check `QUICK_START.md`
2. Review `IMPLEMENTATION_GUIDE.md`
3. Run `test_agent.py`
4. Check logs and error messages

---

**Happy deploying! 🚀**
