# ScholarBase

ScholarBase is the shared literature environment and retrieval backend used by ScholarQuest. It provides RESTful interfaces for semantic retrieval, title matching, paper metadata lookup, and citation-graph queries over a million-scale arXiv collection.

## Attention!

For internal network testing: 172.16.100.204

## Quick Start

**Search papers**

```bash
curl "http://210.45.70.162:4000/paper/search?query=transformer+attention&limit=5"
```

**Get paper by arXiv ID**

```bash
curl "http://210.45.70.162:4000/paper/2309.06180?fields=*"
```

**Get citations**

```bash
curl "http://210.45.70.162:4000/paper/1706.03762/citations?limit=10"
```

## Documentation

Documentation for the underlying API implementation is available at **[ustclewen.github.io/Lewen-API](https://ustclewen.github.io/Lewen-API/)**.

## License

MIT
