"""Mandi (market price) service backed by the data.gov.in Agmarknet API.

Fetches variety-wise daily market prices, normalizes our crop names to the
Agmarknet commodity vocabulary, and caches results to respect the API rate
limit (100 requests/day per key). Queries are state-wide (optionally
commodity-filtered) because Agmarknet's District spellings diverge from
official names; district narrowing happens locally via DISTRICT_ALIASES and a
punctuation-tolerant matcher. Only real market data is returned; when the
live API is unreachable or the key is missing, an empty price list is returned
instead of fabricated MSP numbers.
"""
import asyncio
import time
import httpx
from typing import List, Optional, Dict, Any
from ..config import settings

# Crop alias -> Agmarknet Commodity name
CROP_ALIASES = {
    "wheat": "Wheat",
    "gehun": "Wheat",
    "rice": "Paddy",
    "paddy": "Paddy",
    "dhan": "Paddy",
    "chawal": "Paddy",
    "maize": "Maize",
    "makka": "Maize",
    "mustard": "Mustard",
    "sarson": "Mustard",
}

# Preferred display order for the main crops
CROP_ORDER = {"Wheat": 0, "Paddy": 1, "Maize": 2, "Mustard": 3}

# UI State name -> Agmarknet State spelling (verified against live API).
STATE_ALIASES: Dict[str, str] = {
    "Chhattisgarh": "Chattisgarh",
    "Delhi": "NCT of Delhi",
    "Puducherry": "Pondicherry",
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    # NOTE: Ladakh and DNH-DD have zero records under every spelling — this
    # dataset simply predates/omits them; they fall back to empty prices.
}


def _api_state(state: str) -> str:
    """Translate an official state name to the Agmarknet spelling."""
    for ui_name, api_name in STATE_ALIASES.items():
        if ui_name.lower() == state.lower():
            return api_name
    return state


# UI district name -> Agmarknet District spellings. Agmarknet maintains its own
# (often dated) vocabulary, so official district names frequently never match
# verbatim ("Budaun" vs "Badaun", "Kutch" vs "Kachchh"). Values are verified
# against live API responses; extend this map as new mismatches surface.
DISTRICT_ALIASES: Dict[str, List[str]] = {
    # Uttar Pradesh
    "Ambedkar Nagar": ["Ambedkarnagar"],
    "Bhadohi": ["Sant Ravidas Nagar (Bhadohi)"],
    "Budaun": ["Badaun"],
    "Bulandshahr": ["Bulandshahar"],
    "Chitrakoot": ["Chitrakut"],
    "Farrukhabad": ["Farukhabad"],
    "Gautam Buddha Nagar": ["Gautam Budh Nagar"],
    "Hapur": ["Ghaziabad"],  # Hapur APMC markets are still filed under Ghaziabad
    "Jalaun": ["Jalaun (Orai)"],
    "Kannauj": ["Kannuj"],
    "Kanpur Nagar": ["Kanpur"],
    "Lakhimpur Kheri": ["Lakhimpur", "Khiri (Lakhimpur)"],
    "Mau": ["Mau(Maunathbhanjan)"],
    "Pilibhit": ["Pillibhit"],
    "Raebareli": ["Raebarelli"],
    "Siddharthnagar": ["Siddharth Nagar"],
    # Maharashtra
    "Ahmednagar": ["Ahilyanagar"],
    "Amravati": ["Amarawati"],
    "Aurangabad": ["Chattrapati Sambhajinagar"],
    "Chhatrapati Sambhajinagar": ["Chattrapati Sambhajinagar"],
    "Gondia": ["Gondiya"],
    "Mumbai City": ["Mumbai"],
    "Mumbai Suburban": ["Mumbai"],
    "Osmanabad": ["Dharashiv"],
    # Karnataka
    "Ballari": ["Bellary"],
    "Bengaluru Urban": ["Bengaluru"],
    "Davanagere": ["Davangere"],
    # Gujarat
    "Banaskantha": ["Banaskanth"],
    "Dang": ["The Dangs"],
    "Devbhoomi Dwarka": ["Devbhumi Dwarka"],
    "Junagadh": ["Junagarh"],
    "Kutch": ["Kachchh"],
    "Panchmahal": ["Panchmahals"],
    "Vadodara": ["Vadodara(Baroda)"],
    # Punjab
    "Bathinda": ["Bhatinda"],
    "Fatehgarh Sahib": ["Fatehgarh"],
    "Firozpur": ["Ferozpur"],
    "Rupnagar": ["Ropar (Rupnagar)"],
    "SBS Nagar": ["Nawanshahr"],
    "Tarn Taran": ["Tarntaran"],
    # West Bengal
    "Cooch Behar": ["Coochbehar"],
    "Paschim Medinipur": ["Medinipur(W)"],
    "Purba Medinipur": ["Medinipur(E)"],
    "Purulia": ["Puruliya"],
    "South 24 Parganas": ["Sounth 24 Parganas"],
    # Bihar
    "East Champaran": ["East Champaran/ Motihari", "Purbi Champaran"],
    "Pashchim Champaran": ["Paschim Champaran", "West Champaran"],
    "Saran": ["Chhapra"],
    # Assam / Haryana
    "Kamrup Metropolitan": ["Kamrup Metro"],
    "Nuh": ["Mewat"],
}

# Headline crops always fetched for the home screen
MAIN_CROPS = ["Wheat", "Paddy", "Maize", "Mustard"]

# Some crops are split across several Agmarknet commodity names by region
# ("Paddy" in eastern UP, "Rice"/"Paddy(Common)" in the west), so these are
# fetched under every known name and merged; otherwise whole districts would
# silently lose the crop.
COMMODITY_FAMILY: Dict[str, List[str]] = {
    "Paddy": ["Paddy", "Rice"],
}

# All-India State & UT pick-list (Agmarknet / data.gov.in naming). District lists
# cover every state/UT so a farmer in any state — or one selling in a neighbouring
# state (e.g. a UP farmer trading in Haryana) — can select the right location.
STATE_DISTRICTS = {
    # --- Northern ---
    "Uttar Pradesh": [
        "Agra", "Aligarh", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Ayodhya",
        "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki",
        "Bareilly", "Basti", "Bhadohi", "Bijnor", "Budaun", "Bulandshahr", "Chandauli",
        "Chitrakoot", "Deoria", "Etah", "Etawah", "Farrukhabad", "Fatehpur", "Firozabad",
        "Gautam Buddha Nagar", "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur",
        "Hapur", "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj",
        "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kushinagar",
        "Lakhimpur Kheri", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri",
        "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit",
        "Pratapgarh", "Prayagraj", "Raebareli", "Rampur", "Saharanpur", "Sambhal",
        "Sant Kabir Nagar", "Shahjahanpur", "Shamli", "Shravasti", "Siddharthnagar",
        "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi",
    ],
    "Haryana": [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurugram",
        "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh",
        "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa",
        "Sonipat", "Yamunanagar",
    ],
    "Punjab": [
        "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka",
        "Firozpur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana",
        "Malerkotla", "Mansa", "Moga", "Mohali", "Muktsar", "Pathankot", "Patiala",
        "Rupnagar", "Sangrur", "SBS Nagar", "Tarn Taran",
    ],
    "Himachal Pradesh": [
        "Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kinnaur", "Kullu",
        "Lahaul and Spiti", "Mandi", "Shimla", "Sirmaur", "Solan", "Una",
    ],
    "Uttarakhand": [
        "Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
        "Nainital", "Pauri Garhwal", "Pithoragarh", "Rudraprayag", "Tehri Garhwal",
        "Udham Singh Nagar", "Uttarkashi",
    ],
    "Chandigarh": ["Chandigarh"],
    "Delhi": [
        "Central Delhi", "East Delhi", "New Delhi", "North Delhi", "North East Delhi",
        "North West Delhi", "Shahdara", "South Delhi", "South East Delhi",
        "South West Delhi", "West Delhi",
    ],
    "Jammu and Kashmir": [
        "Anantnag", "Bandipora", "Baramulla", "Budgam", "Doda", "Ganderbal", "Jammu",
        "Kathua", "Kishtwar", "Kulgam", "Kupwara", "Poonch", "Pulwama", "Rajouri",
        "Ramban", "Reasi", "Samba", "Shopian", "Srinagar", "Udhampur",
    ],
    "Ladakh": ["Kargil", "Leh"],

    # --- East ---
    "Bihar": [
        "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Bhojpur",
        "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui",
        "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai",
        "Madhepura", "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada",
        "Pashchim Champaran", "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur",
        "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan", "Supaul", "Vaishali",
    ],
    "Jharkhand": [
        "Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka", "East Singhbhum", "Garhwa",
        "Giridih", "Godda", "Gumla", "Hazaribagh", "Jamtara", "Khunti", "Koderma",
        "Latehar", "Lohardaga", "Pakur", "Palamu", "Ramgarh", "Ranchi", "Sahibganj",
        "Seraikela-Kharsawan", "Simdega", "West Singhbhum",
    ],
    "West Bengal": [
        "Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur",
        "Darjeeling", "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong",
        "Kolkata", "Malda", "Murshidabad", "Nadia", "North 24 Parganas",
        "Paschim Bardhaman", "Paschim Medinipur", "Purba Bardhaman", "Purba Medinipur",
        "Purulia", "South 24 Parganas", "Uttar Dinajpur",
    ],
    "Odisha": [
        "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack",
        "Debagarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur",
        "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Kendujhar", "Khordha",
        "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada",
        "Puri", "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh",
    ],
    "Sikkim": ["East Sikkim", "North Sikkim", "South Sikkim", "West Sikkim"],

    # --- Northeast ---
    "Assam": [
        "Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo", "Chirang",
        "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Dima Hasao", "Goalpara", "Golaghat",
        "Hailakandi", "Hojai", "Jorhat", "Kamrup", "Kamrup Metropolitan", "Karbi Anglong",
        "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", "Nagaon", "Nalbari",
        "Sivasagar", "Sonitpur", "South Salmara-Mankachar", "Tinsukia", "Udalguri",
        "West Karbi Anglong",
    ],
    "Arunachal Pradesh": [
        "Anjaw", "Changlang", "Dibang Valley", "East Kameng", "East Siang", "Kamle",
        "Kra Daadi", "Kurung Kumey", "Lepa Rada", "Lohit", "Longding", "Lower Dibang Valley",
        "Lower Siang", "Lower Subansiri", "Namsai", "Pakke-Kessang", "Papum Pare",
        "Shi-Yomi", "Siang", "Tawang", "Tirap", "Upper Siang", "Upper Subansiri",
        "West Kameng", "West Siang",
    ],
    "Manipur": [
        "Bishnupur", "Chandel", "Churachandpur", "Imphal East", "Imphal West", "Jiribam",
        "Kakching", "Kamjong", "Kangpokpi", "Noney", "Pherzawl", "Senapati", "Tamenglong",
        "Tengnoupal", "Thoubal", "Ukhrul",
    ],
    "Meghalaya": [
        "East Garo Hills", "East Jaintia Hills", "East Khasi Hills", "North Garo Hills",
        "Ri Bhoi", "South Garo Hills", "South West Garo Hills", "South West Khasi Hills",
        "West Garo Hills", "West Jaintia Hills", "West Khasi Hills",
    ],
    "Mizoram": [
        "Aizawl", "Champhai", "Hnahthial", "Khawzawl", "Kolasib", "Lawngtlai", "Lunglei",
        "Mamit", "Saiha", "Saitual", "Serchhip",
    ],
    "Nagaland": [
        "Chumoukedima", "Dimapur", "Kiphire", "Kohima", "Longleng", "Mokokchung", "Mon",
        "Noklak", "Peren", "Phek", "Tuensang", "Wokha", "Zunheboto",
    ],
    "Tripura": [
        "Dhalai", "Gomati", "Khowai", "North Tripura", "Sepahijala", "South Tripura",
        "Unakoti", "West Tripura",
    ],

    # --- West ---
    "Rajasthan": [
        "Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur", "Bhilwara",
        "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur",
        "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu",
        "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Pratapgarh", "Rajsamand",
        "Sawai Madhopur", "Sikar", "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur",
    ],
    "Gujarat": [
        "Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar",
        "Botad", "Chhota Udaipur", "Dahod", "Dang", "Devbhoomi Dwarka", "Gandhinagar",
        "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch", "Mahisagar", "Mehsana",
        "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot",
        "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad",
    ],
    "Goa": ["North Goa", "South Goa"],
    "Dadra and Nagar Haveli and Daman and Diu": [
        "Dadra and Nagar Haveli", "Daman", "Diu",
    ],
    "Maharashtra": [
        "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana",
        "Chandrapur", "Chhatrapati Sambhajinagar", "Dhule", "Gadchiroli", "Gondia",
        "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City",
        "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad",
        "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
        "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
    ],

    # --- Central ---
    "Madhya Pradesh": [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani",
        "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh",
        "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad",
        "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa", "Khargone", "Maihar",
        "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", "Niwari", "Panna",
        "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Seoni",
        "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh",
        "Ujjain", "Umaria", "Vidisha",
    ],
    "Chhattisgarh": [
        "Balod", "Baloda Bazar", "Balrampur", "Bastar", "Bemetara", "Bijapur",
        "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband",
        "Gaurela-Pendra-Marwahi", "Janjgir-Champa", "Jashpur", "Kabirdham", "Kanker",
        "Kondagaon", "Korba", "Koriya", "Mahasamund", "Mungeli", "Narayanpur",
        "Raigarh", "Raipur", "Rajnandgaon", "Sukma", "Surajpur", "Surguja",
    ],

    # --- South ---
    "Andhra Pradesh": [
        "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool",
        "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam", "Vizianagaram",
        "West Godavari", "YSR Kadapa",
    ],
    "Telangana": [
        "Adilabad", "Bhadradri Kothagudem", "Hanumakonda", "Jagtial", "Jangaon",
        "Jayashankar Bhupalpally", "Jogulamba Gadwal", "Kamareddy", "Karimnagar",
        "Khammam", "Kumaram Bheem Asifabad", "Mahabubabad", "Mahabubnagar", "Mancherial",
        "Medak", "Medchal-Malkajgiri", "Mulugu", "Nagarkurnool", "Nalgonda",
        "Narayanpet", "Nirmal", "Nizamabad", "Peddapalli", "Rajanna Sircilla",
        "Ranga Reddy", "Sangareddy", "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy",
        "Warangal", "Yadadri Bhuvanagiri",
    ],
    "Karnataka": [
        "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar",
        "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga",
        "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri",
        "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur",
        "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura",
        "Yadgir",
    ],
    "Tamil Nadu": [
        "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri",
        "Dindigul", "Erode", "Kallakurichi", "Kancheepuram", "Kanyakumari", "Karur",
        "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal",
        "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem",
        "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi",
        "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur",
        "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar",
    ],
    "Kerala": [
        "Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam",
        "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram",
        "Thrissur", "Wayanad",
    ],
    "Puducherry": ["Karaikal", "Mahe", "Puducherry", "Yanam"],
    "Lakshadweep": ["Lakshadweep"],
    "Andaman and Nicobar Islands": [
        "Nicobar", "North and Middle Andaman", "South Andaman",
    ],
}


def _parse_price(raw: str) -> Optional[float]:
    """Parse Agmarknet price strings ('1460', '0', '8540.44') to float."""
    try:
        val = float(raw)
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


class MandiService:
    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = settings.MANDI_CACHE_TTL
        self._fallback_ttl = 300  # retry a failed live API after 5 min, never cache fallback long

    def _cache_key(self, state: str, district: str, commodity: Optional[str], market: Optional[str]) -> str:
        return f"{state}|{district}|{commodity or '*'}|{market or '*'}"

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry:
            # Only full live boards earn the 6h cache; partial/fallback data
            # retries after 5 min so a rate-limited minute doesn't stick.
            ttl = self._cache_ttl if entry[1].get("source") == "live" else self._fallback_ttl
            if time.time() - entry[0] < ttl:
                return entry[1]
        return None

    def _set_cached(self, key: str, data: Dict[str, Any]):
        self._cache[key] = (time.time(), data)

    async def get_prices(
        self,
        crop: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the latest mandi prices, optionally filtered by crop/state/district/market.

        Location is client-driven: state/district come from the farmer's device
        (GPS or manual selection). Config values only serve as last-resort defaults.
        """
        state = (state or settings.MANDI_STATE).strip()
        district = (district or settings.MANDI_DISTRICT).strip()

        # The data.gov.in filters are case-sensitive, so normalize to canonical titles.
        state = self._canonical_state(state)
        district = self._canonical_district(state, district)

        commodity = None
        if crop and crop.strip():
            commodity = CROP_ALIASES.get(crop.strip().lower(), crop.strip().title())

        key = self._cache_key(state, district, commodity, market)
        cached = self._get_cached(key)
        if cached:
            return cached

        data = None
        if settings.MANDI_API_KEY:
            try:
                if commodity:
                    records: List[Dict[str, Any]] = []
                    # One name of the family may legitimately have zero rows
                    # (e.g. bare "Paddy" in western UP) — that must not abort
                    # the remaining names.
                    for name in COMMODITY_FAMILY.get(commodity, [commodity]):
                        try:
                            payload = await self._fetch_live(state, name)
                            records.extend(payload["records"])
                        except Exception as e:
                            print(f"Mandi API fetch failed ({name}): {e}")
                    if not records:
                        raise ValueError(
                            f"No mandi records for {state}/{district} ({commodity})."
                        )
                    prices = self._aggregate(records, None, market, district)
                    data = {
                        "source": "live",
                        "state": state,
                        "district": district,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "prices": prices,
                    }
                else:
                    data = await self._fetch_composite(state, district, market)
            except Exception as e:
                print(f"Mandi API fetch failed: {e}.")

        if not data:
            data = {
                "source": "fallback",
                "state": state,
                "district": district,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prices": [],
            }

        self._set_cached(key, data)
        return data

    async def _fetch_live(self, state: str, commodity: Optional[str], limit: int = 500) -> Dict[str, Any]:
        """Fetch latest records for a state (optionally one commodity).

        District is deliberately NOT sent as an API filter: Agmarknet's District
        vocabulary diverges from official names, so server-side filtering would
        silently drop whole districts. We fetch state-wide and filter locally
        against DISTRICT_ALIASES instead — same request count, no lost data.
        """
        params = {
            "api-key": settings.MANDI_API_KEY,
            "format": "json",
            "limit": limit,
            "filters[State]": _api_state(state),
            "sort[Arrival_Date]": "desc",
        }
        if commodity:
            params["filters[Commodity]"] = commodity

        async with httpx.AsyncClient(
            timeout=12.0,
            headers={"User-Agent": "curl/8.0.1"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(settings.MANDI_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", [])
            if not records:
                raise ValueError(f"No mandi records for {state}.")
            return {"records": records}

    async def _fetch_records(
        self, state: str, commodity: Optional[str], limit: int = 500
    ) -> Optional[List[Dict[str, Any]]]:
        """Per-commodity live fetch with its own cache slot.

        The cache key excludes district because fetches are state-wide; every
        district query in the same state reuses one payload (saves rate limit).
        """
        rkey = f"rec|{state}|{commodity or '*'}"
        entry = self._cache.get(rkey)
        if entry and time.time() - entry[0] < self._cache_ttl:
            return entry[1]
        try:
            payload = await self._fetch_live(state, commodity, limit=limit)
        except Exception as e:
            print(f"Mandi API fetch failed ({commodity or 'general'}): {e}")
            return None
        records = payload["records"]
        self._cache[rkey] = (time.time(), records)
        return records

    async def _fetch_composite(
        self, state: str, district: str, market: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch all main crops + the general latest batch, then aggregate.

        Rate-limit or per-crop failures are tolerated: failed crops are simply
        missing from the response. Only a total failure (every request) returns None.
        """
        records = []
        names: List[Optional[str]] = []
        for crop in MAIN_CROPS:
            names.extend(COMMODITY_FAMILY.get(crop, [crop]))
        names.append(None)  # general latest batch

        # Fire every upstream call in parallel: NIC latency stacks badly when
        # sequential (6 x 30s worst case), and one slow commodity must not
        # starve the rest of the board.
        results = await asyncio.gather(
            *(
                self._fetch_records(state, n, limit=(1000 if n is None else 500))
                for n in names
            ),
            return_exceptions=True,
        )
        successes = 0
        for res in results:
            if isinstance(res, list):
                records.extend(res)
                successes += 1
        if successes == 0:
            return None

        prices = self._aggregate(records, None, market, district)
        # A thin result (rate-limit hit mid-board) must not pin garbage in the
        # 6h live cache — degrade to "partial" so it retries after 5 min.
        source = "live" if successes >= 3 else "partial"
        return {
            "source": source,
            "state": state,
            "district": district,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prices": prices,
        }

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase alphanumeric form, so 'Vadodara(Baroda)' style variants can still match."""
        return "".join(ch for ch in (name or "").lower() if ch.isalnum())

    def _district_matcher(self, district: str):
        """Build a predicate matching Agmarknet District values for a UI district name.

        Matches the exact name, any DISTRICT_ALIASES spelling, or any of those
        after punctuation/space normalization.
        """
        base = district.strip().lower()
        variants = {base}
        for ui_name, aliases in DISTRICT_ALIASES.items():
            if ui_name.lower() == base:
                variants.update(a.strip().lower() for a in aliases)
        normalized = {self._normalize(v) for v in variants}

        def match(record_value: str) -> bool:
            rv = (record_value or "").strip().lower()
            return rv in variants or self._normalize(rv) in normalized

        return match

    def _aggregate(
        self,
        records: List[Dict[str, Any]],
        commodity: Optional[str],
        market: Optional[str],
        district: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Group raw records by (commodity, variety, market), keep latest, compute daily change."""
        match_district = self._district_matcher(district) if district else (lambda _: True)
        groups: Dict[str, Dict[str, Any]] = {}

        for r in records:
            mkt = (r.get("Market") or "").strip()
            if market and market.strip().lower() not in mkt.lower():
                continue
            if not mkt:
                continue
            if not match_district(r.get("District")):
                continue

            commodity_name = (r.get("Commodity") or "").strip()
            variety = (r.get("Variety") or "Other").strip()
            gkey = f"{commodity_name}|{variety}|{mkt}"

            modal = _parse_price(r.get("Modal_Price"))
            entry = groups.get(gkey)
            if entry is None:
                groups[gkey] = {
                    "crop": commodity_name,
                    "variety": variety,
                    "market": mkt,
                    "min_price": _parse_price(r.get("Min_Price")),
                    "max_price": _parse_price(r.get("Max_Price")),
                    "modal_price": modal,
                    "arrival_date": (r.get("Arrival_Date") or "").strip(),
                    "_prev_modal": None,
                }
            else:
                # First record is the latest (sorted desc); second is the previous day
                if entry["_prev_modal"] is None and modal is not None:
                    entry["_prev_modal"] = modal

        prices = []
        for entry in groups.values():
            change = None
            if entry["modal_price"] is not None and entry["_prev_modal"] is not None:
                change = round(entry["modal_price"] - entry["_prev_modal"], 2)
            prices.append({
                "crop": entry["crop"],
                "variety": entry["variety"],
                "market": entry["market"],
                "min_price": entry["min_price"],
                "max_price": entry["max_price"],
                "modal_price": entry["modal_price"],
                "change_per_quintal": change,
                "arrival_date": entry["arrival_date"],
            })

        # Sort: requested/main crops first (by CROP_ORDER), then alphabetically
        prices.sort(key=lambda p: (CROP_ORDER.get(p["crop"], 99), p["crop"], p["market"]))
        return prices


    def list_districts(self, state: Optional[str] = None) -> List[str]:
        """Return the district pick-list for a state (used by the location selector)."""
        state_name = (state or settings.MANDI_STATE).strip()
        key = next(
            (k for k in STATE_DISTRICTS if k.lower() == state_name.lower()),
            None
        )
        return STATE_DISTRICTS.get(key, [])

    def list_states(self) -> List[str]:
        """Return all supported states/UTs (sorted) for the farmer's location picker."""
        return sorted(STATE_DISTRICTS.keys(), key=str.lower)

    def _canonical_state(self, state: str) -> str:
        """Return the canonical casing for a known state, else title-case the input."""
        for k in STATE_DISTRICTS:
            if k.lower() == state.lower():
                return k
        return state.title()

    def _canonical_district(self, state: str, district: str) -> str:
        """Return the Agmarknet-cased district name (case-insensitive match), else title-case."""
        for cand in self.list_districts(state):
            if cand.lower() == district.lower():
                return cand
        return district.title()


mandi_service = MandiService()
