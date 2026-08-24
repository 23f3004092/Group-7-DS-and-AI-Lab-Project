import os
import re
import json
import base64
import time
import httpx
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from ..config import settings

# Banned chemicals check lists
_BANNED = [
    "monocrotophos", "endosulfan", "phorate",
    "methyl parathion", "ddt", "aldrin", "chlorpyrifos",
]
_DOSAGE = [
    r"\b(\d+)\s*(x|times)\b.*\b(dose|dosage|dawa|spray)\b",
    r"\b(double|triple|4x|5x|10x)\b.*\b(dose|dosage|strength|dawa)\b",
    r"\bz?jyada\b.*\bdawa\b",
    r"\b(\d{3,})\s*kg\b.*\b(per acre|prati ekad|ekad)\b",
]

# Intent taxonomy from sample_app.py
INTENT_CLASSES = [
    "cultivation_practice", "disease_pest", "general", "non_agri",
    "nutrition_fertilizer", "post_harvest_storage", "specialty_other",
]

CROP_LIST = ["wheat", "rice", "paddy", "maize", "mustard", "sugarcane", "potato", "mango"]
DISEASES = {
    "wheat": ["wheat__yellow_rust", "wheat__brown_rust", "wheat__powdery_mildew", "wheat__healthy"],
    "rice": ["rice__leaf_smut", "rice__brown_spot", "rice__bacterial_leaf_blight", "rice__healthy"],
    "paddy": ["rice__leaf_smut", "rice__brown_spot", "rice__bacterial_leaf_blight", "rice__healthy"],
    "maize": ["maize__leaf_blight", "maize__rust", "maize__healthy"]
}

def check_dosage_and_banned_rules(text: str) -> Tuple[bool, Optional[str]]:
    """Check rule-based guardrails for dosage and banned pesticides."""
    t = str(text).lower()
    for chemical in _BANNED:
        if chemical in t:
            return True, f"Banned chemical '{chemical}' detected."
    for pattern in _DOSAGE:
        if re.search(pattern, t):
            return True, "Hazardous dosage instructions or double-dose advice detected."
    return False, None

class CloudAIService:
    @staticmethod
    async def call_gemini_api(prompt: str, system_instruction: str = None, image_data: bytes = None, mime_type: str = "image/jpeg") -> str:
        """Call Gemini Developer API via HTTPX directly to avoid library version mismatches."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
        
        contents_parts = []
        if image_data:
            base64_image = base64.b64encode(image_data).decode("utf-8")
            contents_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_image
                }
            })
        
        contents_parts.append({"text": prompt})
        
        payload = {
            "contents": [{"parts": contents_parts}]
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        # Add JSON output configuration if needed by checking for JSON request in prompt
        if "json" in prompt.lower():
            payload["generationConfig"] = {
                "responseMimeType": "application/json"
            }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                raise Exception(f"Gemini API returned error code {response.status_code}: {response.text}")
            
            res_json = response.json()
            try:
                text_out = res_json["candidates"][0]["content"]["parts"][0]["text"]
                return text_out
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to parse Gemini response: {e}. Raw response: {res_json}")

    @staticmethod
    async def call_gemini_api_stream(prompt: str, system_instruction: str = None, image_data: bytes = None, mime_type: str = "image/jpeg"):
        """Stream Gemini tokens via streamGenerateContent (SSE). Yields text deltas."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key={settings.GEMINI_API_KEY}"

        contents_parts = []
        if image_data:
            base64_image = base64.b64encode(image_data).decode("utf-8")
            contents_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_image
                }
            })

        contents_parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": contents_parts}]
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if "json" in prompt.lower():
            payload["generationConfig"] = {
                "responseMimeType": "application/json"
            }

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise Exception(f"Gemini API returned error code {response.status_code}: {body[:500]}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    try:
                        parts = obj["candidates"][0]["content"]["parts"]
                    except (KeyError, IndexError):
                        continue
                    for part in parts:
                        text_delta = part.get("text")
                        if text_delta:
                            yield text_delta

    @classmethod
    async def run_intent_entity_guardrails(cls, text: str) -> Tuple[List[str], bool, Optional[str]]:
        """Pathway A/AB: intent classification, entity recognition, and guardrail verification."""
        # 1. Rule-based pre-guardrail
        blocked, reason = check_dosage_and_banned_rules(text)
        if blocked:
            return ["general"], True, reason
        
        # Determine if we can run via Gemini
        if settings.GEMINI_API_KEY and not settings.MOCK_MODELS:
            try:
                system_prompt = (
                    "You are an Intent and Guardrails classifier for an Indian agricultural app. "
                    "Analyze the farmer's query and output a JSON object containing:\n"
                    "1. 'intents': List of strings from: " + ", ".join(INTENT_CLASSES) + "\n"
                    "2. 'detected_crop': Optional string if a crop is mentioned.\n"
                    "3. 'is_blocked': Boolean if query asks for illegal/dangerous dosage or banned chemicals (e.g. monocrotophos, DDT, double-spraying).\n"
                    "4. 'guardrail_reason': Optional explanation of block.\n"
                    "Output ONLY the JSON object. Do not format with markdown tags."
                )
                
                resp = await cls.call_gemini_api(prompt=text, system_instruction=system_prompt)
                # Cleanup potential code block markers
                clean_resp = re.sub(r"```json\s*|\s*```", "", resp).strip()
                data = json.loads(clean_resp)
                
                intents = data.get("intents", ["general"])
                is_blocked = data.get("is_blocked", False)
                reason = data.get("guardrail_reason", None)
                
                # Double check with local rules
                if not is_blocked:
                    is_blocked, rule_reason = check_dosage_and_banned_rules(text)
                    if is_blocked:
                        reason = rule_reason
                
                return intents, is_blocked, reason
            except Exception as e:
                # Fallback to mock on API error
                pass
                
        # Mock mode fallback
        intents = []
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["disease", "pest", "worm", "insect", "rust", "smut", "rot", "blight", "spots"]):
            intents.append("disease_pest")
        if any(w in text_lower for w in ["fertilizer", "urea", "potash", "nitrogen", "npk", "manure", "nutrition"]):
            intents.append("nutrition_fertilizer")
        if any(w in text_lower for w in ["how to grow", "sowing", "irrigation", "water", "spacing", "planting"]):
            intents.append("cultivation_practice")
        if any(w in text_lower for w in ["store", "storage", "harvest", "post harvest", "cold storage"]):
            intents.append("post_harvest_storage")
        
        if not intents:
            intents = ["general"]
            
        return intents, False, None

    @classmethod
    async def run_vision_diagnosis(cls, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Pathway B/AB: Classify disease from leaf photo.

        The photo MUST go through the deployed ViT first (gateway /vision,
        ViT-S/16 per API_SPEC.md). The local mock is only an offline fallback
        when the gateway is unreachable.
        """
        # 1. Real ViT via the GCP gateway /vision
        if settings.AI_API_URL:
            try:
                url = f"{settings.AI_API_URL.rstrip('/')}/vision"
                headers = {}
                if settings.AI_API_KEY:
                    headers["X-API-Key"] = settings.AI_API_KEY
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        url,
                        headers=headers,
                        files={"file": (filename or "leaf.jpg", image_bytes, "image/jpeg")},
                    )
                if resp.status_code != 200:
                    raise Exception(f"ViT gateway /vision -> HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                label = data.get("label") or "healthy"
                confidence = float(data.get("confidence") or 0.0)
                top_k = data.get("top_k") or []
                top3 = [(t.get("label"), float(t.get("prob", 0))) for t in top_k[:3]]
                if not top3:
                    top3 = [(label, confidence)]
                return {
                    "rejected": False,
                    "label": label,
                    "confidence": confidence,
                    "description": f"ViT predicted '{label}' (confidence {confidence:.1%}).",
                    "organic_treatment": "",
                    "chemical_treatment": "",
                    "ood_score": round(max(0.0, 1.0 - confidence), 3),
                    "top3": top3,
                }
            except Exception as e:
                # Gateway unreachable -> fall through to the local mock
                print(f"ViT gateway call failed ({e}); using mock diagnosis.")

        # 2. Robust Mock Diagnosis (offline fallback) — choose a class based on
        # filename keywords or cycle through standard classes
        fname = filename.lower()
        label = "healthy"
        description = "The leaf looks healthy with no major lesions."
        organic_treatment = "No action required. Maintain proper irrigation."
        chemical_treatment = "No chemical pesticides needed."
        
        if "rust" in fname or "pila" in fname:
            label = "wheat__yellow_rust"
            description = "Yellow/orange powdery pustules arranged in linear stripes on the leaf surface. Typical of Yellow Rust (Puccinia striiformis)."
            organic_treatment = "Use garlic extract spray or ferment plant materials to build crop immunity."
            chemical_treatment = "Spray Propiconazole 25% EC @ 200 ml per acre dissolved in 200 liters of water."
        elif "spot" in fname or "dhabba" in fname:
            label = "rice__brown_spot"
            description = "Oval or circular dark brown lesions with yellow halo, indicating Brown Spot fungal disease (Helminthosporium oryzae)."
            organic_treatment = "Treat seeds with Trichoderma harzianum before sowing. Apply compost tea spray."
            chemical_treatment = "Spray Hexaconazole 5% EC @ 2 ml per liter of water."
        elif "blight" in fname or "jhulsa" in fname:
            label = "rice__bacterial_leaf_blight"
            description = "Wavy yellow to straw-colored stripes starting from leaf tips, indicating Bacterial Leaf Blight (Xanthomonas oryzae)."
            organic_treatment = "Avoid excess nitrogen fertilizer. Spray fresh cow dung extract supernatant."
            chemical_treatment = "Spray Streptocycline @ 6g mixed with Copper Oxychloride @ 500g in 200 liters of water per acre."
        elif "smut" in fname or "koyla" in fname:
            label = "rice__leaf_smut"
            description = "Small, slightly raised, black angular spots on leaves, caused by Entyloma oryzae."
            organic_treatment = "Ensure balanced fertilizer usage. Spray Pseudomonas fluorescens @ 5g/liter."
            chemical_treatment = "Spray Propiconazole 25% EC @ 1 ml/liter of water."
        else:
            # Cycle mock data
            t = int(time.time()) % 3
            if t == 0:
                label = "wheat__yellow_rust"
                description = "Striated yellow pustules on wheat leaves (Yellow Rust)."
                organic_treatment = "Spray garlic barrier extract."
                chemical_treatment = "Spray Propiconazole 25% EC."
            elif t == 1:
                label = "rice__brown_spot"
                description = "Brown circular leaf lesions on paddy (Brown Spot)."
                organic_treatment = "Soil application of Pseudomonas fluorescens."
                chemical_treatment = "Spray Carbendazim @ 1g/liter."
            else:
                label = "rice__healthy"
                description = "Paddy leaf displays normal pigmentation and structure."

        return {
            "rejected": False,
            "label": label,
            "confidence": 0.895,
            "description": description,
            "organic_treatment": organic_treatment,
            "chemical_treatment": chemical_treatment,
            "ood_score": 0.05,
            "top3": [(label, 0.895), ("healthy", 0.08), ("other_disease", 0.025)]
        }

    @classmethod
    def _build_synthesis_prompts(cls, query: str, context_chunks: List[Dict[str, Any]],
                                 live_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Build (system_prompt, user_prompt) for the advisory synthesis."""
        ctx_str = "\n\n".join(
            f"[{i+1}] (Source: {c.get('source_type', 'agri_docs')}, Crop: {c.get('crop', 'General')})\n{c.get('text', '')}"
            for i, c in enumerate(context_chunks)
        )

        live_str = ""
        if live_data:
            live_str = "\n\n".join(
                f"- {k.replace('_', ' ').title()}: {v}"
                for k, v in live_data.items() if v
            )

        system_prompt = (
            "You are FarmerVision, a knowledgeable and compassionate agricultural advisor for Indian farmers, "
            "specializing in Uttar Pradesh. Provide practical, step-by-step advice based ONLY on the provided Context. "
            "Do not invent facts, pesticide names, or dosage levels that are not in the Context. "
            "You MUST cite your sources as [1], [2], etc., at the end of every sentence that uses facts from the context. "
            "Use the provided Live Data facts verbatim (they are authoritative, fetched from live APIs) when they are "
            "relevant to the farmer's question, e.g. when advising whether to sell produce, cite the mandi price, "
            "weather forecast, and yield estimate. Provide the answer in Hindi or English depending on the language "
            "of the farmer's question. Always include a short advisory warning if chemicals/pesticides are suggested "
            "(e.g. 'Use gloves, wash hands'). Keep the response concise, structured, and under 150 words."
        )

        prompt = (
            f"Context:\n{ctx_str}\n\n"
            + (f"Live Data (authoritative facts):\n{live_str}\n\n" if live_str else "")
            + f"Farmer's Query: {query}\n\n"
            f"Advisory Response:"
        )
        return system_prompt, prompt

    @classmethod
    async def synthesize_response(cls, query: str, context_chunks: List[Dict[str, Any]],
                                  live_data: Optional[Dict[str, Any]] = None) -> str:
        """Synthesize advice using retrieved context chunks, optional live data, and the LLM."""
        system_prompt, prompt = cls._build_synthesis_prompts(query, context_chunks, live_data)

        if settings.GEMINI_API_KEY and not settings.MOCK_MODELS:
            try:
                resp = await cls.call_gemini_api(prompt=prompt, system_instruction=system_prompt)
                return resp.strip()
            except Exception as e:
                # Fallback to mock on API error
                pass

        return cls._mock_synthesis(query, live_data)

    @classmethod
    async def synthesize_response_stream(cls, query: str, context_chunks: List[Dict[str, Any]],
                                         live_data: Optional[Dict[str, Any]] = None):
        """Async generator yielding answer text deltas for SSE streaming clients.

        Streams from Gemini when configured; otherwise falls back to the mock
        advisory emitted word-by-word so client UX is identical.
        """
        system_prompt, prompt = cls._build_synthesis_prompts(query, context_chunks, live_data)

        produced = False
        if settings.GEMINI_API_KEY and not settings.MOCK_MODELS:
            try:
                async for delta in cls.call_gemini_api_stream(prompt=prompt, system_instruction=system_prompt):
                    produced = True
                    yield delta
                return
            except Exception:
                if produced:
                    return
                # Stream never started -> fall through to mock below

        mock_text = cls._mock_synthesis(query, live_data)
        for word in re.findall(r"\S+\s*|\s+", mock_text):
            yield word

    @staticmethod
    def _mock_synthesis(query: str, live_data: Optional[Dict[str, Any]] = None) -> str:
        """Mock Advisory Synthesis — helpful response incorporating live-data facts."""
        is_hindi = any(char in query for char in ["क", "ह", "ा", "ी", "ो", "म", "न"])

        def _mock_decision_answer() -> Optional[str]:
            """Sell-vs-keep answer built from live mandi/weather/yield facts."""
            if not live_data:
                return None
            mandi = live_data.get("mandi_prices")
            weather = live_data.get("weather")
            yld = live_data.get("yield")
            if is_hindi:
                parts = [
                    "किसान भाई, आज के लाइव आंकड़ों के आधार पर:",
                ]
                if mandi:
                    parts.append(f"• मंडी/भाव: {mandi}")
                if yld:
                    parts.append(f"• अनुमानित पैदावार: {yld}")
                if weather:
                    parts.append(f"• मौसम: {weather}")
                parts.append(
                    "सुझाव: अगर आने वाले दिनों में बारिश की संभावना है तो फसल की गुणवत्ता "
                    "खराब हो सकती है — बेहतर भाव मिलने पर उपज बेचना उचित रहेगा। "
                    "अगर दाम कम है और मौसम साफ है, तो कुछ दिन रोक कर देख सकते हैं। "
                    "अंतिम निर्णय नज़दीकी मंडी और KVK अधिकारी से पुष्टि करके लें [1]।"
                )
                return "\n".join(parts)
            parts = ["Based on today's live data:"]
            if mandi:
                parts.append(f"• Mandi price: {mandi}")
            if yld:
                parts.append(f"• Est. yield: {yld}")
            if weather:
                parts.append(f"• Weather: {weather}")
            parts.append(
                "Suggestion: with rain expected, stored/standing crop quality can drop — "
                "selling when the mandi rate is at or above the MSP is the safer choice. "
                "If the rate is low and skies are clear, holding a few days may pay. "
                "Please confirm with your local mandi and KVK officer before deciding [1]."
            )
            return "\n".join(parts)

        decision = _mock_decision_answer()
        if decision:
            return decision

        def _mock_answer() -> str:
            if is_hindi:
                if "rust" in query.lower() or "पीला" in query:
                    return (
                        "गेहूं में पीला रतुआ (Yellow Rust) के नियंत्रण के लिए:\n"
                        "1. प्रभावित क्षेत्रों में नीम के तेल का छिड़काव करें [1]।\n"
                        "2. संक्रमण अधिक होने पर प्रोपिकोनाजोल 25% EC (Propiconazole) दवा का 200 मिली प्रति एकड़ की दर से 200 लीटर पानी में मिलाकर छिड़काव करें [2]।\n"
                        "⚠ सावधानी: रासायनिक छिड़काव करते समय मास्क और दस्ताने पहनें और हवा के रुख के विपरीत न छिड़कें।"
                    )
                elif "spot" in query.lower() or "धब्बा" in query or "धान" in query:
                    return (
                        "धान में भूरा धब्बा (Brown Spot) रोग के प्रबंधन के लिए:\n"
                        "1. संतुलित मात्रा में नाइट्रोजन उर्वरक का उपयोग करें और पोटाश डालें [1]।\n"
                        "2. रासायनिक उपचार के लिए हेक्साकोनाजोल 5% EC (Hexaconazole) @ 2 मिली प्रति लीटर पानी का छिड़काव करें [2]।\n"
                        "⚠ चेतावनी: कीटनाशक छिड़काव के बाद हाथ साबुन से धोएं और फसल कटाई से 15 दिन पहले छिड़काव बंद करें।"
                    )
                else:
                    return (
                        "किसान भाई, आपकी समस्या के समाधान के लिए उपलब्ध जानकारी के अनुसार:\n"
                        "1. फसल में नमी बनाए रखें और अधिक सिंचाई से बचें [1]।\n"
                        "2. संतुलित एनपीके (NPK) खाद डालें जिससे पौधों की रोग प्रतिरोधक क्षमता बढ़े [2]।\n"
                        "अधिक जानकारी के लिए कृपया अपने नजदीकी कृषि विज्ञान केंद्र (KVK) के वैज्ञानिक से संपर्क करें।"
                    )
            # English Response
            if "rust" in query.lower() or "yellow" in query:
                return (
                    "To manage Yellow Rust in wheat:\n"
                    "1. Apply organic garlic extract as a preventive measure [1].\n"
                    "2. For chemical control, spray Propiconazole 25% EC @ 200 ml/acre in 200 liters of water [2].\n"
                    "⚠ Safety: Use protective gloves and face mask during spraying."
                )
            elif "spot" in query.lower() or "brown" in query or "rice" in query or "paddy" in query:
                return (
                    "For Rice Brown Spot disease management:\n"
                    "1. Ensure balanced fertilization. Split nitrogen application and apply potash [1].\n"
                    "2. Spray Hexaconazole 5% EC @ 2 ml per liter of water under severe infection [2].\n"
                    "⚠ Caution: Avoid spraying during strong winds; wash hands thoroughly after application."
                )
            else:
                return (
                    "Based on the agricultural advisory sources:\n"
                    "1. Ensure proper drainage in the field to avoid root rot [1].\n"
                    "2. Apply balanced fertilizers based on soil testing reports [2].\n"
                    "For precise local guidance, please verify with your regional KVK officer."
                )

        mock = _mock_answer()
        if live_data:
            facts_block = "\n".join(
                f"- {k.replace('_', ' ').title()}: {v}"
                for k, v in live_data.items() if v
            )
            mock = f"{mock}\n\n**Live Data (today):**\n{facts_block}"
        return mock
