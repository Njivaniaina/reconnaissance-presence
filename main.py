#!/usr/bin/env python3
import os
import time
import requests
import cv2
import numpy as np
import insightface
import csv
import re
import psycopg2
from psycopg2 import sql

# ---------------- CONFIG ----------------
HOST = "192.168.10.181:5001"
USER = "mit"
PWD  = "123456"
SHOT_URL = f"http://{HOST}/shot.jpg"
INTERVAL_SECONDS = 5
PROFILES_DIR = "profile"
OUTPUT_DIR = "captures"
LOG_FILE = os.path.join(OUTPUT_DIR, "recognition_log.csv")
THRESHOLD = 0.40

# Postgres config — ADAPTE ÇA à ton environnement
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB   = "mit"
PG_USER = "admin"
PG_PWD  = "123456"
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Charger InsightFace ----------
print("Chargement du modèle InsightFace ...")
model = insightface.app.FaceAnalysis(name="buffalo_l")
try:
    model.prepare(ctx_id=0, det_size=(640, 640))
    print("Modèle chargé sur GPU (ctx_id=0).")
except Exception as e:
    print("GPU non dispo / erreur, utilisation CPU. Détails:", e)
    model.prepare(ctx_id=-1, det_size=(640, 640))
    print("Modèle chargé sur CPU (ctx_id=-1).")

def get_first_face_embedding_from_bgr(img_bgr):
    if img_bgr is None:
        return None, None
    faces = model.get(img_bgr)
    if not faces:
        return None, None
    face = faces[0]
    emb = face.embedding
    bbox = face.bbox.astype(int).tolist()
    return emb, bbox

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# -------- Charger profils et pré-calcul des embeddings --------
profiles = []  # {"id": id, "name": filename_base, "emb": emb, "path": fpath}
id_pattern = re.compile(r'profile[-_](.+)', re.IGNORECASE)  # supporte profile-123 ou profile_123

if os.path.isdir(PROFILES_DIR):
    for fname in sorted(os.listdir(PROFILES_DIR)):
        fpath = os.path.join(PROFILES_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = cv2.imread(fpath)
        if img is None:
            print(f"[Profil] impossible de lire {fpath}, skip.")
            continue
        emb, bbox = get_first_face_embedding_from_bgr(img)
        if emb is None:
            print(f"[Profil] aucun visage détecté dans {fname}, skip.")
            continue
        base = os.path.splitext(fname)[0]
        # extraire ID depuis le nom : profile-<ID>.jpg -> on prend <ID>
        m = id_pattern.match(base)
        profile_id = None
        if m:
            profile_id = m.group(1)
        else:
            # si ne correspond pas, on utilise le nom entier comme ID
            profile_id = base
        profiles.append({"id": profile_id, "name": base, "emb": emb, "path": fpath})
        print(f"[Profil] chargé: {base} -> id={profile_id}")
else:
    print(f"Le dossier '{PROFILES_DIR}' n'existe pas.")

if not profiles:
    print("⚠️ Aucun profil valide chargé. Toutes les personnes seront marquées 'Inconnu'.")

def match_profile(emb, profiles_list):
    if emb is None or not profiles_list:
        return None, 0.0
    best_name = None
    best_score = -1.0
    best_id = None
    for p in profiles_list:
        score = cosine_similarity(emb, p["emb"])
        if score > best_score:
            best_score = score
            best_name = p["name"]
            best_id = p["id"]
    return best_id, best_name, best_score

def fetch_shot(url, user=None, pwd=None, timeout=8):
    try:
        if user and pwd:
            resp = requests.get(url, auth=(user, pwd), timeout=timeout)
        else:
            resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            img_arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        else:
            print(f"HTTP {resp.status_code} lors de la requête {url}")
            return None
    except Exception as e:
        print("Erreur lors de la requête:", e)
        return None

# ---------- Connexion à la BDD PostgreSQL ----------
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PWD
        )
        return conn
    except Exception as e:
        print("Impossible de se connecter à la base de données:", e)
        return None

def lookup_person_by_id(person_id):
    """Retourne 'Lastname Firstname' ou None si introuvable ou erreur."""
    if person_id is None:
        return None
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            # la table s'appelle "Person" et la colonne id est text
            q = sql.SQL('SELECT lastname, firstname FROM "Person" WHERE id = %s')
            cur.execute(q, (person_id,))
            row = cur.fetchone()
            if row:
                lastname, firstname = row
                if firstname:
                    return f"{firstname} {lastname}"
                else:
                    return f"{lastname}"
            else:
                return None
    except Exception as e:
        print("Erreur lors de la requête DB:", e)
        return None
    finally:
        conn.close()

# ---------- Initialiser fichier de log ----------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf, delimiter=';')
        writer.writerow(["Horodatage", "Fichier", "Résultat", "Similarité", "Matched_ID", "Person_Name"])
    print(f"Fichier de log créé: {LOG_FILE}")

def log_recognition(timestamp, fname, result, similarity, matched_id, person_name):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf, delimiter=';')
        writer.writerow([timestamp, fname, result, f"{similarity:.3f}", matched_id or "", person_name or ""])

# ------------------ boucle principale ------------------
print("Démarrage des captures périodiques (Ctrl+C pour arrêter)...")
count = 0
try:
    while True:
        img = fetch_shot(SHOT_URL, user=USER, pwd=PWD)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        file_time = time.strftime("%Y%m%d_%H%M%S")

        if img is None:
            print(f"[{timestamp}] Impossible de récupérer l'image.")
            time.sleep(2)
            continue

        count += 1
        emb, bbox = get_first_face_embedding_from_bgr(img)
        annotated = img.copy()

        result = "Aucun visage"
        score = 0.0
        matched_id = None
        person_name = None

        if emb is not None:
            matched_id, matched_name, score = match_profile(emb, profiles)
            # lookup person name in DB if matched_id exists
            if matched_id is not None:
                person_name = lookup_person_by_id(matched_id)
            if matched_id is not None and score >= THRESHOLD:
                # afficher ID et nom si disponible
                if person_name:
                    result = f"{matched_id} - {person_name}"
                else:
                    result = f"{matched_id} - {matched_name}"
                color = (0, 255, 0)
            else:
                result = f"Inconnu ({score:.3f})"
                color = (0, 165, 255)

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                pos = (x1, max(20, y1 - 10))
            else:
                pos = (10, 30)
            cv2.putText(annotated, result, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        else:
            # aucun visage détecté -> on marque "Aucun visage"
            cv2.putText(annotated, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Enregistrement temporaire du fichier
        fname = os.path.join(OUTPUT_DIR, f"capture_{file_time}_{count}.jpg")
        cv2.imwrite(fname, annotated)

        # Si aucun visage détecté --> supprimer le fichier local sauvegardé
        if emb is None:
            try:
                os.remove(fname)
                # log avec précision que le fichier a été supprimé
                log_recognition(timestamp, os.path.basename(fname), "Aucun visage - fichier supprimé", 0.0, "", "")
                print(f"[{count}] {timestamp} - Aucun visage -> fichier {os.path.basename(fname)} supprimé localement.")
            except Exception as e:
                # si erreur suppression, on loggue quand même
                log_recognition(timestamp, os.path.basename(fname), "Aucun visage - suppression échouée", 0.0, "", "")
                print(f"[{count}] {timestamp} - Aucun visage, mais suppression échouée: {e}")
        else:
            # On garde le fichier et on log la correspondance
            # matched_id peut être None si pas de profil chargé
            log_recognition(timestamp, os.path.basename(fname), result, score, matched_id, person_name)
            print(f"[{count}] {timestamp} - {result} (score={score:.3f}) -> sauvegardé: {os.path.basename(fname)}")

        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("Arrêt demandé par l'utilisateur.")
except Exception as ex:
    print("Erreur inattendue:", ex)
finally:
    print("Terminé.")

