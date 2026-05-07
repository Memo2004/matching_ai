from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI()


model = SentenceTransformer('all-mpnet-base-v2')

# -------------------------
# Database
# -------------------------
def fetch_mentors_from_db():
    conn = None
    try:
        conn = psycopg2.connect(
            host="ep-icy-mouse-a4rz6n2f-pooler.us-east-1.aws.neon.tech",
            port=5432,
            database="neondb",
            user="neondb_owner",
            password="npg_UBNtL0bo8MVQ",
            sslmode="require"
        )

        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT
                mp.id AS "mentorId",
                u."firstName",
                u."lastName",
                u.bio,
                mp."jobTitle",
                mp."yearsOfExperience",
                ms."skillName",
                ms."level"
            FROM public."MentorProfile" mp
            JOIN public."User" u ON mp."userId" = u.id
            LEFT JOIN public."MentorSkill" ms ON ms."mentorId" = mp.id
            WHERE mp."isAvailable" = true;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        mentors_dict = {}

        for row in rows:
            mentor_id = row['mentorId']

            if mentor_id not in mentors_dict:
                mentors_dict[mentor_id] = {
                    "mentorId": mentor_id,
                    "name": f"{row['firstName']} {row['lastName']}",
                    "bio": row['bio'],
                    "jobTitle": row['jobTitle'],
                    "yearsOfExperience": row['yearsOfExperience'],
                    "skills": []
                }

            if row['skillName']:
                mentors_dict[mentor_id]["skills"].append({
                    "skillName": row['skillName'],
                    "level": row['level']
                })

        cursor.close()
        return list(mentors_dict.values())

    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

# -------------------------
# Helpers
# -------------------------
def extract_skills(skill_list):
    return [s["skillName"].lower() for s in skill_list]

def normalize_skill(skill):
    return skill.lower().replace("5", "")

def clean_text(text):
    if not text:
        return ""
    return text.lower().replace("\n", " ").strip()

def project_to_text(p):
    return f"""
    Project: {p.project_name}
    Description: {p.description}
    Required skills: {' '.join(p.skills)}
    Tools: {' '.join(p.tools)}
    Difficulty: {p.difficulty}
    """

def mentor_to_text(m):
    skills = [normalize_skill(s) for s in extract_skills(m["skills"])]
    bio = clean_text(m["bio"])
    job = m['jobTitle'] if m['jobTitle'] else "mentor"

    return f"""
    Mentor job: {job}
    Experience: {m['yearsOfExperience']} years
    Skills: {' '.join(skills)}
    Bio: {bio}
    """

def cosine_similarity(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return F.cosine_similarity(a, b, dim=0).item()

def skill_overlap(project_skills, mentor_skills):
    mentor_skills = [normalize_skill(s) for s in mentor_skills]
    return len(set(project_skills) & set(mentor_skills))

# -------------------------
# Request Schemas
# -------------------------
class ProjectRequest(BaseModel):
    project_name: str
    description: str
    difficulty: str
    skills: list[str]
    tools: list[str]

class StudentRequest(BaseModel):
    student_id: str
    skills: list[str]
    learning_goals: list[str]
    topics: list[str]
    preferred_difficulty: str
    learning_mode: str

# -------------------------
# Text builders
# -------------------------
def student_to_text(s: StudentRequest):
    return f"""
    Student skills: {' '.join(s.skills)}
    Learning goals: {' '.join(s.learning_goals)}
    Topics of interest: {' '.join(s.topics)}
    Preferred difficulty: {s.preferred_difficulty}
    Learning mode: {s.learning_mode}
    """

def score_mentors(query_embedding, mentors, query_skills: list[str]):
    mentor_texts = [mentor_to_text(m) for m in mentors]
    mentor_embeddings = model.encode(mentor_texts)

    results = []
    for i, m in enumerate(mentors):
        sim = cosine_similarity(query_embedding, mentor_embeddings[i])

        mentor_skills = extract_skills(m["skills"])
        overlap = skill_overlap(query_skills, mentor_skills)
        overlap = overlap / len(query_skills) if query_skills else 0

        final_score = (sim * 0.7) + (overlap * 0.3)

        results.append({
            "mentorId": m["mentorId"],
            "name": m["name"],
            "jobTitle": m["jobTitle"],
            "score": round(final_score, 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# -------------------------
# Routes
# -------------------------

@app.get("/")
def home():
    return {"message": "API is working "}

@app.get("/test-db")
def test_db():
    mentors = fetch_mentors_from_db()

    if isinstance(mentors, dict):
        return mentors

    return {
        "count": len(mentors),
        "sample": mentors[:1]
    }

@app.post("/match")
def match_project(project: ProjectRequest):
    mentors = fetch_mentors_from_db()

    if isinstance(mentors, dict):
        return mentors

    if not mentors:
        return {"error": "No mentors found"}

    project_embedding = model.encode(project_to_text(project))
    query_skills = [normalize_skill(s) for s in project.skills]

    return score_mentors(project_embedding, mentors, query_skills)[:3]

@app.post("/match/student")
def match_student(student: StudentRequest):
    mentors = fetch_mentors_from_db()

    if isinstance(mentors, dict):
        return mentors

    if not mentors:
        return {"error": "No mentors found"}

    student_embedding = model.encode(student_to_text(student))
    query_skills = [normalize_skill(s) for s in student.skills + student.learning_goals]

    return score_mentors(student_embedding, mentors, query_skills)