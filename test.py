def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    print(f"num_chunks: {num_chunks}")
    chunk_size = token_count // num_chunks
    print(f"chunk_size: {chunk_size}")
    remaining_tokens = token_count % token_limit
    print(f"remaining_tokens: {remaining_tokens}")
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks
        print(f"chunk_size: {chunk_size}")

    return chunk_size


print(calculate_chunk_size(1200, 500))
