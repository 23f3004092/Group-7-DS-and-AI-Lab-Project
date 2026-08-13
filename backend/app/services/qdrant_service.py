import time
import random
from typing import List, Dict, Any, Tuple
from ..config import settings

# Sample mock documents based on real FarmerVision agricultural corpus (representing Policy & KCC logs)
MOCK_DOCS = [
    # Wheat Yellow Rust (KCC/PDF)
    {
        "id": "doc_wheat_1",
        "text": "Yellow Rust (Pila Ratua) in wheat is caused by Puccinia striiformis. Symptoms include yellow stripe pustules on leaves. For biological control, use neem-based formulations. For chemical control, spray Propiconazole 25% EC @ 200 ml mixed with 200 liters of water per acre.",
        "crop": "wheat",
        "source_type": "pdf_policy",
        "year": 2024,
        "page": 12
    },
    {
        "id": "doc_wheat_2",
        "text": "KCC query 4821: Farmer reported yellow dust on wheat leaves in Jhansi district. Expert advised immediate application of Propiconazole 25% EC @ 1ml/liter. Warned against delay as disease spreads quickly under high humidity.",
        "crop": "wheat",
        "source_type": "kcc_log",
        "year": 2023,
        "page": None
    },
    # Rice Brown Spot (KCC/PDF)
    {
        "id": "doc_rice_1",
        "text": "Brown Spot (Bhura Dhabba) in rice/paddy is a fungal disease showing oval, dark brown spots on leaves. It is associated with nutrient-deficient soils. Farmers should apply balanced fertilizer including potash. Under high severity, spray Hexaconazole 5% EC @ 2 ml per liter of water.",
        "crop": "rice",
        "source_type": "pdf_policy",
        "year": 2022,
        "page": 45
    },
    {
        "id": "doc_rice_2",
        "text": "KCC query 9324: Rice crop leaf spots reported in Varanasi. Expert suggested spraying Carbendazim @ 1g/liter and ensuring adequate potash fertilization during top-dressing.",
        "crop": "rice",
        "source_type": "kcc_log",
        "year": 2023,
        "page": None
    },
    # PM-KISAN & KCC Policy (PDF)
    {
        "id": "doc_policy_1",
        "text": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) scheme provides an income support of Rs. 6000 per year in three equal installments to all landholding farmer families across the country. Application requires Aadhaar, land registry documents, and bank account details.",
        "crop": "general",
        "source_type": "pdf_policy",
        "year": 2024,
        "page": 2
    },
    {
        "id": "doc_policy_2",
        "text": "Kisan Credit Card (KCC) scheme provides farmers with timely access to credit for crop cultivation, post-harvest expenses, and maintenance. Interest subvention of 2% is available, with an additional 3% subvention for prompt repayment, reducing the effective interest rate to 4%.",
        "crop": "general",
        "source_type": "pdf_policy",
        "year": 2023,
        "page": 8
    },
    # Cultivation & Fertilizer practice
    {
        "id": "doc_fertilizer_1",
        "text": "Paddy nitrogen management guidelines: Total urea requirement for rice is 120 kg per hectare, applied in three split doses: 50% at transplanting, 25% at tillering, and 25% at panicle initiation. Excessive urea application increases vulnerability to pests and blast disease.",
        "crop": "rice",
        "source_type": "pdf_policy",
        "year": 2024,
        "page": 18
    },
    {
        "id": "doc_fertilizer_2",
        "text": "Wheat fertilizer dosage guidelines for Uttar Pradesh: Recommended N:P:K ratio is 120:60:40 kg per hectare. Apply full doses of Phosphorus and Potash along with half Nitrogen during sowing. Apply remaining Nitrogen after first and second irrigation.",
        "crop": "wheat",
        "source_type": "pdf_policy",
        "year": 2023,
        "page": 14
    }
]

class QdrantService:
    def __init__(self):
        self.client = None
        self.embedder = None
        self.initialized = False
        
        # We try to connect to Qdrant, but do not fail if it isn't running
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            
            # Use a quick timeout so we don't hang if server is down
            self.client = QdrantClient(url=settings.QDRANT_URL, timeout=3.0)
            # Try to get collection to check connection
            self.client.get_collection(settings.COLLECTION_NAME)
            
            # Load embedding model (only if Qdrant is connected)
            # Download will take time on first run, so we wrap it
            self.embedder = SentenceTransformer(settings.BGE_MODEL_ID, device="cpu")
            self.initialized = True
            print(f"Qdrant connected successfully on {settings.QDRANT_URL} with {settings.BGE_MODEL_ID} embedder.")
        except Exception as e:
            print(f"Qdrant initialization bypassed/failed ({e}). Running in LOCAL RETRIEVAL MOCK mode.")
            self.client = None
            self.embedder = None
            self.initialized = False

    def retrieve(self, query: str, intents: List[str] = None, current_settings: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Search documents. In mockup mode, it uses keyword overlap and intent-weighting.
        In production, it queries Qdrant server and applies weights.
        
        Returns:
            hits: list of matched chunks with score, text, and metadata
            tier: "grounded" | "fallback" | "abstain"
            top_score: raw float score of the top match
        """
        t_grounded = current_settings.get("tier_grounded", settings.TIER_GROUNDED) if current_settings else settings.TIER_GROUNDED
        t_fallback = current_settings.get("tier_fallback", settings.TIER_FALLBACK) if current_settings else settings.TIER_FALLBACK
        
        # Retrieve using real Qdrant + SentenceTransformer if loaded
        if self.initialized and self.client and self.embedder:
            try:
                # Real search logic
                q_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
                
                # Apply multi-query or intent filters if needed
                # For simplicity, search once
                results = self.client.query_points(
                    collection_name=settings.COLLECTION_NAME,
                    query=q_vec,
                    limit=settings.TOP_K,
                    with_payload=True
                ).points
                
                hits = []
                for point in results:
                    payload = point.payload
                    score = point.score
                    
                    # Apply intent-based source weighting
                    source_type = payload.get("source_type", "kcc_log")
                    weight = 1.0
                    
                    # Read weights
                    w_pdf_pol = current_settings.get("weight_pdf_policy", settings.WEIGHT_PDF_POLICY) if current_settings else settings.WEIGHT_PDF_POLICY
                    w_kcc_pol = current_settings.get("weight_kcc_policy", settings.WEIGHT_KCC_POLICY) if current_settings else settings.WEIGHT_KCC_POLICY
                    w_pdf_prac = current_settings.get("weight_pdf_practice", settings.WEIGHT_PDF_PRACTICE) if current_settings else settings.WEIGHT_PDF_PRACTICE
                    w_kcc_prac = current_settings.get("weight_kcc_practice", settings.WEIGHT_KCC_PRACTICE) if current_settings else settings.WEIGHT_KCC_PRACTICE
                    
                    if intents and "policy" in intents:
                        weight = w_pdf_pol if source_type == "pdf_policy" else w_kcc_pol
                    elif intents and "disease_pest" in intents:
                        weight = w_pdf_prac if source_type == "pdf_policy" else w_kcc_prac
                        
                    weighted_score = score * weight
                    
                    hits.append({
                        "id": str(point.id),
                        "score": score,  # keep raw score for tier classification
                        "weighted_score": weighted_score,
                        "text": payload.get("text", ""),
                        "crop": payload.get("crop", "general"),
                        "district": payload.get("district", ""),
                        "source_type": source_type,
                        "page": payload.get("page", None),
                        "year": payload.get("year", None)
                    })
                
                # Sort by weighted score
                hits = sorted(hits, key=lambda x: x["weighted_score"], reverse=True)
                
                if not hits:
                    return [], "abstain", 0.0
                    
                top_score = hits[0]["score"] # Confidence is computed on the raw score, never the weighted one! (§9.6)
                tier = "grounded" if top_score >= t_grounded else "fallback" if top_score >= t_fallback else "abstain"
                return hits, tier, top_score
                
            except Exception as e:
                print(f"Error querying Qdrant: {e}. Falling back to mock search.")
                
        # Mock / Keyword search fallback
        query_words = set(query.lower().split())
        hits = []
        
        for doc in MOCK_DOCS:
            text = doc["text"].lower()
            overlap = sum(1 for w in query_words if w in text or (doc["crop"] in query_words))
            if overlap == 0:
                continue
                
            # Base similarity score
            score = 0.4 + (overlap / (len(query_words) + 5)) * 0.5
            score = min(score, 0.95)
            
            # Apply weights
            source_type = doc["source_type"]
            weight = 1.0
            w_pdf_pol = current_settings.get("weight_pdf_policy", settings.WEIGHT_PDF_POLICY) if current_settings else settings.WEIGHT_PDF_POLICY
            w_kcc_pol = current_settings.get("weight_kcc_policy", settings.WEIGHT_KCC_POLICY) if current_settings else settings.WEIGHT_KCC_POLICY
            w_pdf_prac = current_settings.get("weight_pdf_practice", settings.WEIGHT_PDF_PRACTICE) if current_settings else settings.WEIGHT_PDF_PRACTICE
            w_kcc_prac = current_settings.get("weight_kcc_practice", settings.WEIGHT_KCC_PRACTICE) if current_settings else settings.WEIGHT_KCC_PRACTICE
            
            if intents:
                if any(i in ["policy", "nutrition_fertilizer"] for i in intents):
                    weight = w_pdf_pol if source_type == "pdf_policy" else w_kcc_pol
                elif "disease_pest" in intents:
                    weight = w_pdf_prac if source_type == "pdf_policy" else w_kcc_prac
            
            weighted_score = score * weight
            
            hits.append({
                "id": doc["id"],
                "score": score,
                "weighted_score": weighted_score,
                "text": doc["text"],
                "crop": doc["crop"],
                "source_type": doc["source_type"],
                "page": doc["page"],
                "year": doc["year"]
            })
            
        hits = sorted(hits, key=lambda x: x["weighted_score"], reverse=True)
        
        # If no hits, generate random small score to fallback
        if not hits:
            # Let's see if query is general farming
            if any(w in query.lower() for w in ["farming", "crop", "urea", "pest", "disease", "kisan"]):
                # Return general doc with fallback score
                doc = MOCK_DOCS[4] # PM-KISAN
                score = 0.58
                hits.append({
                    "id": doc["id"],
                    "score": score,
                    "weighted_score": score,
                    "text": doc["text"],
                    "crop": doc["crop"],
                    "source_type": doc["source_type"],
                    "page": doc["page"],
                    "year": doc["year"]
                })
            else:
                return [], "abstain", 0.35
                
        top_score = hits[0]["score"]
        tier = "grounded" if top_score >= t_grounded else "fallback" if top_score >= t_fallback else "abstain"
        return hits, tier, top_score

# Singleton instance
qdrant_service = QdrantService()
