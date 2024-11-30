def chunk_corpus(corpus: list, chunk_size: int = 64) -> list:
    """
    Chunk the corpus into smaller parts. Run the following command to download the required nltk data:
    python -c "import nltk; nltk.download('punkt')"

    @param corpus: the formatted corpus, see README.md
    @param chunk_size: the size of each chunk, i.e., the number of words in each chunk
    @return: chunked corpus, a list
    """
    from nltk.tokenize import sent_tokenize, word_tokenize

    new_corpus = []
    for p in corpus:
        text = p['text']
        idx = p['idx'] if 'idx' in p else p['_id']
        title = p['title']

        sentences = sent_tokenize(text)
        current_chunk = []
        current_chunk_size = 0

        chunk_idx = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            if current_chunk_size + len(words) > chunk_size:
                new_corpus.append({
                    **p,
                    'title': title,
                    'text': " ".join(current_chunk),
                    'idx': idx + f"_{chunk_idx}",
                })
                current_chunk = words
                current_chunk_size = len(words)
                chunk_idx += 1
            else:
                current_chunk.extend(words)
                current_chunk_size += len(words)

        if current_chunk:  # there are still some words left
            new_corpus.append({
                **p,
                'title': title,
                'text': " ".join(current_chunk),
                'idx': f"{idx}_{chunk_idx}",
            })

    return new_corpus


def merge_chunk_scores(id_score: dict):
    """
    Merge the scores of chunks into the original passage
    @param id_score: a dictionary of passage_id (the chunk id, str) -> score (float)
    @return: a merged dictionary of passage_id (the original passage id, str) -> score (float)
    """
    merged_scores = {}
    for passage_id, score in id_score.items():
        passage_id = passage_id.split('_')[0]
        if passage_id not in merged_scores:
            merged_scores[passage_id] = 0
        merged_scores[passage_id] = max(merged_scores[passage_id], score)
    return merged_scores


def merge_chunks(corpus: list):
    """
    Merge the chunks of a corpus into the original passage
    @param corpus: a passage list
    @return: a merged corpus, dict
    """

    new_corpus = {}
    for p in corpus:
        idx = p['idx']
        if '_' not in idx:
            new_corpus[idx] = p
        else:
            original_idx = idx.split('_')[0]
            if original_idx not in new_corpus:
                new_corpus[original_idx] = {
                    **p,
                    'text': p['text'],
                    'idx': original_idx,
                }
            else:
                new_corpus[original_idx]['text'] += ' ' + p['text']

    return list(new_corpus.values())


def generate_hash(input_string, algorithm='sha224'):
    import hashlib
    try:
        algo = getattr(hashlib, algorithm)()
    except AttributeError:
        raise ValueError(f'Unsupported algorithm: {algorithm}')
    algo.update(input_string.encode('utf-8'))
    return algo.hexdigest()


def check_continuity(data):
    sorted_values = [v for k, v in data.items()]

    breaks = []
    continuous_ranges = []
    start = 0

    for i in range(1, len(sorted_values)):
        if sorted_values[i] != sorted_values[i - 1] + 1:
            breaks.append(i)
            continuous_ranges.append((sorted_values[start], sorted_values[i - 1]))
            start = i

    continuous_ranges.append((sorted_values[start], sorted_values[-1]))

    if len(breaks) > 0:
        print(f"Breaks at indices: {breaks}")
        print(f"Number of continuous subarrays: {len(continuous_ranges)}")
        print(f"Continuous ranges (start, end): {continuous_ranges}")
        exit(1)


def convert_html_to_markdown(html_content):
    import html2text
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    markdown_text = converter.handle(html_content)
    return markdown_text


def split_html_to_text_segments(html_content, max_words_per_segment=128):
    # Parse HTML content to extract pure text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    pure_text = soup.get_text(separator=" ", strip=True)
    pure_text = " ".join(pure_text.split())

    # Split text into sentences
    import nltk
    sentences = nltk.sent_tokenize(pure_text)

    # Group sentences into segments, ensuring each segment does not exceed the max word limit
    segments = []
    current_segment = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If adding the current sentence exceeds the max word limit, finalize the current segment
        if current_word_count + sentence_word_count > max_words_per_segment:
            segments.append(" ".join(current_segment))
            current_segment = [sentence]
            current_word_count = sentence_word_count
        else:
            current_segment.append(sentence)
            current_word_count += sentence_word_count

    # Add the last segment if it contains any sentences
    if current_segment:
        segments.append(" ".join(current_segment))

    return segments
