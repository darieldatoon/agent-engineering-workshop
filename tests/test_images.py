from kea.claims.images import photo_from_content, text_from_content


def test_photo_from_studio_upload():
    content = [
        {"type": "text", "text": "P-1001, KA 1182, bag worth $400"},
        {"type": "image", "source_type": "base64", "mime_type": "image/png", "data": "AAAA"},
    ]
    assert photo_from_content(content) == "data:image/png;base64,AAAA"
    assert text_from_content(content) == "P-1001, KA 1182, bag worth $400"
    assert photo_from_content("no photo here") is None
