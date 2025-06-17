def parse_chroma_metadata(metadata: dict) -> dict:
    """
    Parse the metadata for the Chroma database query
    Args:
        metadata: Metadata of the text
                    E.g. `{"url": "https://example.com", "title": "example.pdf", "therapeutic_area": "oncology", "timestamp": "06-2025"}`
    Return:
        Metadata with operator and value
                E.g. `{"timestamp": {"$eq": "06-2025"}}`
    """
    parsed_metadata = {}
    for key, value in metadata.items():
        parsed_metadata[key] = {"$eq": value}
    return parsed_metadata
