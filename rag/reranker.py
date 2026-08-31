from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ms-marco-MiniLM-L-6-v2 is trained on general web search relevance, not
# pharmacology -- confirmed by testing that for a warfarin+fluconazole
# query, it ranked a chunk about an unrelated drug (nefazodone, matched on
# generic "highly protein bound" phrasing) ABOVE the chunk that actually
# names both warfarin and fluconazole with the specific bleeding/
# prothrombin-time interaction (raw scores 5.36 vs 4.87 -- a narrow, common
# margin). A small boost per literal drug-name match fixes exactly this
# failure mode without overriding genuine semantic relevance: typical score
# spread between clearly-relevant and clearly-irrelevant content is 5-10
# points, so this only breaks near-ties, not one-sided calls.
NAME_MATCH_BOOST = 1.0

def rerank(query, docs, boost_terms=None):
    pairs = [(query, d.page_content) for d in docs]
    scores = list(model.predict(pairs))

    if boost_terms:
        terms = [t.lower() for t in boost_terms if t]
        for i, doc in enumerate(docs):
            content_lower = doc.page_content.lower()
            matches = sum(1 for t in terms if t in content_lower)
            scores[i] += matches * NAME_MATCH_BOOST

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:2]]