# Matching AI - Mentor & Project Matcher

A high-performance FastAPI application that uses the **Cohere Multilingual Embedding API** to match mentors with projects based on semantic similarity and skill overlap.

## 🚀 Features
- **Semantic Matching:** Uses Cohere's `embed-multilingual-v3.0` to understand the context of project descriptions and mentor bios.
- **Skill Overlap Logic:** Combines AI similarity (70%) with direct skill matching (30%) for precise results.
- **PostgreSQL Integration:** Fetches real-time mentor profiles and skill sets from a relational database.
- **Dockerized:** Optimized lightweight container (~150MB) for fast deployment.
- **Auto-generated API Docs:** Built-in Swagger UI via FastAPI.

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Framework:** FastAPI
- **AI Engine:** Cohere API
- **Database:** PostgreSQL (psycopg2)
- **Deployment:** Docker

## 📋 Prerequisites
- Docker installed.
- A Cohere API Key (get one at [dashboard.cohere.com](https://dashboard.cohere.com/)).
- A PostgreSQL database with `MentorProfile`, `User`, and `MentorSkill` tables.

## ⚙️ Environment Variables
Create a `.env` file in the root directory with the following variables:
```env
DB_HOST=your_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_user
DB_PASSWORD=your_password
COHERE_API_KEY=your_cohere_key

 🐳 Docker Usage
- Build the Image
Bash
docker build -t matching-ai .
Run the Container Locally
Bash
docker run -d -p 8000:8000 --env-file .env matching-ai
Pull from Docker Hub
If you want to use the pre-built image:

Bash
docker pull memo412/matching-ai:latest
🔌 API Endpoints
1. Home / Health Check
GET /
Returns the status of the API and the engine being used.

2. Match Project
POST /match
Request Body Example:

JSON
{
    "project_name": "E-commerce App",
    "description": "Building a mobile app for selling clothes using React Native.",
    "difficulty": "Intermediate",
    "skills": ["React Native", "JavaScript"],
    "tools": ["VS Code", "Expo"]
}
📄 License
MIT License


---

### How to upload to GitHub:

1.  **Initialize Git** (if you haven't already):
    ```bash
    git init
    ```
2.  **Add your files**:
    ```bash
    git add .
    ```
3.  **Commit your changes**:
    ```bash
    git commit -m "Initial commit: Added Cohere AI matching and README"
    ```
4.  **Create a new repository on GitHub.com** and then link it:
    ```bash
    git remote add origin https://github.com/memo2004/matching_ai.git
    git branch -M main
    git push -u origin main
    ```

**Important Reminder:** Ensure your `.gitignore` file includes `.env` so you don't accidentally push your database passwords or Cohere API keys to GitHub!