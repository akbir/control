def format_key_suffix(key_suffix):
    if key_suffix:
        if key_suffix[0] != "_":
            key_suffix = f"_{key_suffix}"
    else:
        key_suffix = ""
    return key_suffix
