from matchers.embedding.antique_embedding_matcher import AntiqueEmbeddingMatcher


def main():
  matcher = AntiqueEmbeddingMatcher()

  while True:
    query_text = input("Enter your query (or 'q' to quit): ")
    if query_text.lower() == 'q':
      break

    suggested_queries = matcher.get_similar_queries(query_text)

    print("Suggested Queries:")
    if suggested_queries:
      for query in suggested_queries:
        print(f"\t- {query}")
    else:
      print("\t- No similar queries found.")

    print("\n")


if __name__ == "__main__":
  main()
