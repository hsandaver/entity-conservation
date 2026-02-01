---
title: Entity Explorer
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Entity Explorer

An interactive web application for exploring, analyzing, and annotating entity relationships in bibliographic and linked data.

## Features

- **Multiple Data Format Support**: Import from RDF (TTL, RDF, NT), JSON-LD, MARC (binary & text), and RIS formats
- **Interactive Graph Visualization**: Explore entity relationships with interactive network graphs
- **Entity Search & Filtering**: Find and filter entities by type, label, and relationships
- **Rich-Text Annotations**: Add and edit annotations for entities with formatting support
- **Entity Matching**: Fuzzy matching and deduplication of entities
- **Export Capabilities**: Export graph data in multiple formats

## Getting Started

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/hsandaver/entity-conservation.git
cd entity-explorer-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Deployed Version

Visit the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/hsandaver/entityexplorer)

## Supported Formats

- **RDF**: Turtle (.ttl), RDF/XML (.rdf), N-Triples (.nt)
- **JSON-LD**: Standard JSON-LD format
- **MARC**: Binary (.mrc) and text (.mrk) formats
- **RIS**: RIS bibliographic format (.ris)

## Architecture

- **Frontend**: Streamlit with custom HTML components
- **Backend**: Python-based data processing and graph analysis
- **Data Storage**: In-memory session state management
- **Graph Processing**: NetworkX for graph algorithms and analysis

## Configuration

Environment variables (optional):
- `AWS_ACCESS_KEY_ID`: For S3 storage integration
- `AWS_SECRET_ACCESS_KEY`: For S3 storage integration

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
