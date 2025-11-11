# ---------------------------------
# 🐳 DOCKERFILE — Telco Churn Dashboard
# ---------------------------------

# 1️⃣ Use Python 3.10 (same as your project)
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy project files into container
COPY . /app

# 4️⃣ Install system dependencies (for matplotlib, seaborn, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 5️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Expose Streamlit’s default port
EXPOSE 8501

# 7️⃣ Run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
