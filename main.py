import os
import math
import cohere
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize Cohere client
COHERE_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_KEY)

# -------------------------
# Database Logic
# -------------------------
def fetch_mentors_from_db():
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require"
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                mp.id AS "mentorId", u."firstName", u."lastName", u.bio,
                mp."jobTitle", mp."yearsOfExperience", ms."skillName", ms."level"
            FROM public."MentorProfile" mp
            JOIN public."User" u ON mp."userId" = u.id
            LEFT JOIN public."MentorSkill" ms ON ms."mentorId" = mp.id
            WHERE mp."isAvailable" = true;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        mentors_dict = {}
        for row in rows:
            m_id = row['mentorId']
            if m_id not in mentors_dict:
                mentors_dict[m_id] = {
                    "mentorId": m_id,
                    "name": f"{row['firstName']} {row['lastName']}",
                    "bio": row['bio'],
                    "jobTitle": row['jobTitle'],
                    "yearsOfExperience": row['yearsOfExperience'],
                    "skills": []
                }
            if row['skillName']:
                mentors_dict[m_id]["skills"].append({"skillName": row['skillName'], "level": row['level']})
        
        cursor.close()
        return list(mentors_dict.values())
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn: conn.close()

# -------------------------
# Helpers
# -------------------------
def normalize_skill(skill):
    return skill.lower().replace("5", "").strip()

def project_to_text(p):
    return f"Project: {p.project_name}. Description: {p.description}. Skills: {', '.join(p.skills)}. Tools: {', '.join(p.tools)}."

def mentor_to_text(m):
    skills = [normalize_skill(s["skillName"]) for s in m["skills"]]
    return f"Job: {m['jobTitle'] or 'Mentor'}. Skills: {', '.join(skills)}. Bio: {m['bio'] or ''}."

def cosine_similarity(v1, v2):
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x**2 for x in v1))
    mag2 = math.sqrt(sum(x**2 for x in v2))
    return dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0

# -------------------------
# Schema & Routes
# -------------------------
class ProjectRequest(BaseModel):
    project_name: str
    description: str
    difficulty: str
    skills: list[str]
    tools: list[str]

@app.get("/")
def home():
    return {"status": "online", "engine": "Cohere API"}

@app.post("/match")
def match_project(project: ProjectRequest):
    mentors = fetch_mentors_from_db()
    if not mentors or isinstance(mentors, dict):
        return mentors if mentors else {"error": "No mentors available"}

    # Convert project and mentors to text format
    p_text = project_to_text(project)
    m_texts = [mentor_to_text(m) for m in mentors]
    
    # Batch request to Cohere for Embeddings
    response = co.embed(
        texts=[p_text] + m_texts,
        model='embed-multilingual-v3.0',
        input_type='search_query'
    )
    
    embeddings = response.embeddings
    project_emb = embeddings[0]
    mentor_embs = embeddings[1:]

    results = []
    project_skills_set = set(s.lower() for s in project.skills)

    for i, m in enumerate(mentors):
        # 1. Semantic Similarity Score (AI)
        sim_score = cosine_similarity(project_emb, mentor_embs[i])
        
        # 2. Skill Overlap Score (Keyword matching)
        m_skills_set = set(s['skillName'].lower() for s in m['skills'])
        overlap = len(project_skills_set & m_skills_set)
        overlap_score = overlap / len(project_skills_set) if project_skills_set else 0
        
        # Weighted final score (70% AI / 30% Keywords)
        final_score = (sim_score * 0.7) + (overlap_score * 0.3)
        
        results.append({
            "mentorId": m["mentorId"],
            "name": m["name"],
            "score": round(final_score, 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]