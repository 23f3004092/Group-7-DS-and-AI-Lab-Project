import sys

# Force UTF-8 on Windows so Devanagari in payloads doesn't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME = "agri_knowledge"

def main():
    print("=" * 60)
    print("QDRANT METADATA VERIFICATION")
    print("=" * 60)
    
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=10)
        info = client.get_collection(COLLECTION_NAME)
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {info.points_count:,}")
    except Exception as e:
        print(f"ERROR connecting to Qdrant or getting collection: {e}")
        sys.exit(1)

    print("\nQuerying 10,000 random KCC chunks to verify metadata schemas...")
    
    # Filter for KCC source_type
    kcc_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(value="kcc")
            )
        ]
    )

    # Use the scroll API to grab a large sample without vector search overhead
    records, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=kcc_filter,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )

    if not records:
        print("No KCC records found!")
        sys.exit(1)

    print(f"Sampled {len(records):,} KCC records.")
    
    # Track statistics
    fields_present = {
        "crop": 0, "district": 0, "year": 0, "month": 0, 
        "season": 0, "query_type": 0, "category": 0, "language": 0
    }
    
    print("\n--- Payload Schema Analysis ---")
    for r in records:
        payload = r.payload
        for f in fields_present.keys():
            # Check if field exists and is not None
            if f in payload and payload[f] is not None:
                fields_present[f] += 1

    total = len(records)
    for field, count in fields_present.items():
        pct = (count / total) * 100
        print(f"  {field:<15}: present in {count:>5,} / {total:<5,} ({pct:>5.1f}%)")

    print("\n--- Example Raw Payload ---")
    import json
    print(json.dumps(records[0].payload, indent=2, ensure_ascii=True))
    
    print("\nDone.")

if __name__ == "__main__":
    main()
